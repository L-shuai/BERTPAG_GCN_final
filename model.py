import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast


class REModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(
            args.model_name_or_path, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = nn.CrossEntropyLoss()
        self.seg = PAG(self.encoder, hidden_size=hidden_size)
        self.fc1 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(3 * hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(hidden_size, args.num_class)
        )

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None, se=None, oe=None, pos1=None, pos2=None, mask=None, ):
        E, T, S = self.seg(pos1, pos2, mask, input_ids, attention_mask, ss, os, se, oe)
        E = self.fc1(E)
        T = self.fc2(T)
        S = self.fc3(S)
        # h = torch.cat((E,T, S), dim=-1)
        h = torch.cat((E, S), dim=-1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs


class PAG(nn.Module):
    def __init__(self,encoder, lambda_pcnn=0.05, word_dim=768, lambda_san=1.0, pos_dim=32, pos_len=100, hidden_size=768, dropout_rate=0.5):
        super(PAG, self).__init__()
        self.embedding = Entity_Aware_Embedding(encoder, pos_dim, pos_len)
        self.PCNN = PCNN(word_dim, pos_dim, lambda_pcnn, hidden_size)
        self.SAN = SAN(word_dim, pos_dim, lambda_san)

        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(3 * word_dim, 3 * hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X_Pos1, X_Pos2, X_Mask,input_ids=None, attention_mask=None, ss=None, os=None, se=None, oe=None):
        # Embed
        E, T, Xp, Xe = self.embedding(X_Pos1, X_Pos2,input_ids=input_ids, attention_mask=attention_mask, ss=ss, os=os, se=se, oe=oe)
        # Encode
        S = self.PCNN(Xp, Xe, X_Mask)
        U = self.SAN(Xp, Xe)
        # gate
        X = self.selective_gate(S, U)
        # X = torch.cat((S,U), dim=-1)
        return E, T, X

    def selective_gate(self, S, U):
        G = torch.sigmoid(self.fc2(torch.tanh(self.fc1(U))))
        X = G * S
        return X

# EntityAware
class Entity_Aware_Embedding(nn.Module):
    def __init__(self, encoder, pos_dim=32, pos_len=150):
        super(Entity_Aware_Embedding, self).__init__()
        self.encoder = encoder # bert

        self.pos1_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)

    def forward(self,X_Pos1, X_Pos2, input_ids=None, attention_mask=None, ss=None, os=None, se=None, oe=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0] 
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        type_ss_emb = pooled_output[idx, ss - 1]
        os_emb = pooled_output[idx, os]
        type_os_emb = pooled_output[idx, os - 1]
        se_emb = pooled_output[idx, se]
        oe_emb = pooled_output[idx, oe]
        E = torch.cat((ss_emb, se_emb, os_emb, oe_emb), dim=-1)
        T = torch.cat((type_ss_emb, type_os_emb), dim=-1)
        Xp = self.word_pos_embedding(pooled_output, X_Pos1, X_Pos2)
        Xe = self.word_ent_embedding(pooled_output, idx,ss, os, se, oe)     
        return E, T, Xp, Xe

    def word_pos_embedding(self,pooled_output,X_Pos1, X_Pos2):
        X_Pos1 = self.pos1_embedding(X_Pos1)
        X_Pos2 = self.pos2_embedding(X_Pos2)
        Xp = torch.cat([pooled_output, X_Pos1, X_Pos2], -1)
        return Xp

    def word_ent_embedding(self, pooled_output, idx, ss=None, os=None, se=None, oe=None):
        X_Ent1 = pooled_output[idx, ss].unsqueeze(1).expand(pooled_output.shape)
        X_Ent2 = pooled_output[idx, os].unsqueeze(1).expand(pooled_output.shape)
        Xe = torch.cat([pooled_output, X_Ent1, X_Ent2], -1)
        return Xe

# PCNN
class PCNN(nn.Module):
    def __init__(self, word_dim, pos_dim, lam, hidden_size=768):
        super(PCNN, self).__init__()
        mask_embedding = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.cnn = nn.Conv1d(3 * word_dim, hidden_size, 3, padding=1)
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(2 * pos_dim + word_dim, 3 * word_dim)
        self.hidden_size = hidden_size
        self.lam = lam

    def forward(self, Xp, Xe, X_mask):
        A = torch.sigmoid((self.fc1(Xe / self.lam)))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X = self.pool(X, X_mask)
        X = torch.tanh(X)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, self.hidden_size * 3)

# self-attention
class SAN(nn.Module):
    def __init__(self, word_dim, pos_dim, lam):
        super(SAN, self).__init__()
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(2 * pos_dim + word_dim, 3 * word_dim)
        self.fc1_att = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2_att = nn.Linear(3 * word_dim, 3 * word_dim)
        self.lam = lam

    def forward(self, Xp, Xe):
        # embedding
        A = torch.sigmoid((self.fc1(Xe / self.lam)))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        # print(X.shape)
        # encoder
        A = self.fc2_att(torch.tanh(self.fc1_att(X)))
        P = torch.softmax(A, 1)
        X = torch.sum(P * X, 1)
        return X
