import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast
from layers.dynamic_rnn import DynamicLSTM
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class REModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(
            args.model_name_or_path, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = nn.CrossEntropyLoss()
        # self.loss_fnt = nn.MSELoss()
        self.seg = PAG(self.encoder, hidden_size=hidden_size)
        self.fc1 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(3 * hidden_size, hidden_size)
        # self.classifier = nn.Sequential(
        #     nn.Linear(3 * hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(p=args.dropout_prob),
        #     nn.Linear(hidden_size, args.num_class)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(hidden_size, args.num_class)
        )
        self.classifier_gcn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(hidden_size, args.num_class)
        )
        self.gc1 = GraphConvolution(768, 768)
        # self.gc1 = GraphConvolution(768*2, 768*2)
        self.gc2 = GraphConvolution(768, 768)
        self.gc3 = GraphConvolution(768, 768)
        self.gc4 = GraphConvolution(768, 768)
        self.gc5 = GraphConvolution(768, 768)
        self.gc6 = GraphConvolution(768, 768)
        self.gc7 = GraphConvolution(768, 768)




    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None, se=None, oe=None, pos1=None, pos2=None, mask=None,matrix=None ):
        #E:64*3072  S:64*2304  T:64*1536  text_out:64*180*2304
        E, T, S ,text_out= self.seg(pos1, pos2, mask, input_ids, attention_mask, ss, os, se, oe)
        E = self.fc1(E) #E:64*1536  Embedding
        T = self.fc2(T)#T::64*768   type_ss_emb  没有用到
        S = self.fc3(S)#S:64*768   门控机制(融合PCNN和Attnetion)
        # h = torch.cat((E,T, S), dim=-1)
        h = torch.cat((E, S), dim=-1)#h:64*2304
        # print("matrix.type",type(matrix),str(type(matrix)))
        text_len = torch.sum(input_ids != 0, dim=-1).cpu()
        # print("text_len=",text_len,"matrix_len=",matrix.shape[0])
        if "None" in str(type(matrix)):
            print("*"*100)
            # text_out_shape_0 = text_out.shape[0]
            # # matrix = np.zeros((180, 180)).astype('float32')
            # matrix = np.ones((text_out_shape_0,180, 180)).astype('float32')
            # # matrix = torch.from_numpy(matrix)
            # matrix = torch.tensor(matrix)
            h = torch.cat((E, S), dim=-1)
            logits = self.classifier(h)
            # logits_probs = torch.sigmoid(logits) # mapping 0 - 1
            outputs = (logits,)
            # outputs = (logits_probs,)
            if labels is not None:
                loss = self.loss_fnt(logits.float(), labels)
                # loss = self.loss_fnt(logits_probs.float(), labels)
                outputs = (loss,) + outputs
            return outputs
        else:
            # 1
            gcn_output = self.gc1(text_out,matrix)# gcn_output:[64, 180, 768]
            gcn_output = F.relu(gcn_output) #x:[64, 180, 2304])
            # 2
            gcn_output = self.gc1(gcn_output, matrix)  # gcn_output:[64, 180, 768]
            gcn_output = F.relu(gcn_output)  # x:[64, 180, 2304])
            # # # 3
            gcn_output = self.gc1(gcn_output, matrix)  # gcn_output:[64, 180, 768]
            gcn_output = F.relu(gcn_output)  # x:[64, 180, 2304])
            # # #4
            gcn_output = self.gc1(gcn_output, matrix)  # gcn_output:[64, 180, 768]
            gcn_output = F.relu(gcn_output)  # x:[64, 180, 2304])
            #
            # #5
            gcn_output = self.gc1(gcn_output, matrix)  # gcn_output:[64, 180, 768]
            gcn_output = F.relu(gcn_output)  # x:[64, 180, 2304])
            #
            # #6
            # gcn_output = self.gc6(gcn_output, matrix)  # gcn_output:[64, 180, 768]
            # gcn_output = F.relu(gcn_output)  # x:[64, 180, 2304])
            #
            # #7
            # gcn_output = self.gc7(gcn_output, matrix)  # gcn_output:[64, 180, 768]
            # gcn_output = F.relu(gcn_output)  # x:[64, 180, 2304])

            x = self.mask(gcn_output, ss,os,se,oe)#x:[64, 180, 768]
            alpha_mat = torch.matmul(gcn_output, text_out.transpose(1, 2))#alpha_mat:([64, 180, 180])
            alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)#alpha([64, 1, 180])
            x = torch.matmul(alpha, text_out).squeeze(1)#x:([64, 2304])
            # print("h:",h.shape)
            # print("x:",x.shape)
            final_rep = torch.cat((E,S+x),dim=-1)
            # final_rep = torch.cat((x,S),dim=-1)
            # final_rep = h+x
            # final_rep = torch.cat((S, x), dim=-1)#final_rep([64, 1536])
            # logits = self.classifier_gcn(final_rep)#torch.Size([64, 10])
            logits = self.classifier(final_rep)#torch.Size([64, 10])
            # logits = self.classifier(h)#torch.Size([64, 10])
            # logits = self.classifier_gcn(final_rep)#torch.Size([64, 10])
            # logits_probs = torch.sigmoid(logits)  # mapping 0 - 1
            outputs = (logits,)
            # outputs = (logits_probs,)
            if labels is not None:
                loss = self.loss_fnt(logits.float(), labels)
                # loss = self.loss_fnt(logits_probs.float(), labels)
                outputs = (loss,) + outputs
            return outputs



        # h = torch.cat((E, S), dim=-1)
        # logits = self.classifier(h)
        # outputs = (logits,)
        # if labels is not None:
        #     loss = self.loss_fnt(logits.float(), labels)
        #     outputs = (loss,) + outputs
        # return outputs


    def mask(self, x, ss,os,se,oe):
        batch_size, seq_len = x.shape[0], x.shape[1]
        ss = ss.cpu().numpy()
        os = os.cpu().numpy()
        se = se.cpu().numpy()
        oe = oe.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            start = min(ss[i],se[i],os[i],oe[i])
            end = min(ss[i],se[i],os[i],oe[i])
            for j in range(start):
                mask[i].append(0)
            for j in range(start, end+1):
                mask[i].append(1)
            for j in range(end+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(0)
        return mask*x

class PAG(nn.Module):
    def __init__(self,encoder, lambda_pcnn=0.05, word_dim=768, lambda_san=1.0, pos_dim=32, pos_len=100, hidden_size=768, dropout_rate=0.5):
        super(PAG, self).__init__()
        self.embedding = Entity_Aware_Embedding(encoder, pos_dim, pos_len)
        self.PCNN = PCNN(word_dim, pos_dim, lambda_pcnn, hidden_size)
        self.SAN = SAN(word_dim, pos_dim, lambda_san)

        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(3 * word_dim, 3 * hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        # self.text_lstm = DynamicLSTM(768, 1152, num_layers=1, batch_first=True, bidirectional=True)
        self.text_lstm = DynamicLSTM(768, 768, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, X_Pos1, X_Pos2, X_Mask,input_ids=None, attention_mask=None, ss=None, os=None, se=None, oe=None):
        # Embed  E:64*3072  T:64*1536  pooled_output:([64, 180, 768])  Xp:   Xe:64*180*2304 X_Pos1:64*180    input_ids: 64*180   attention_mask:64*180
        E, T, Xp, Xe,pooled_output = self.embedding(X_Pos1, X_Pos2,input_ids=input_ids, attention_mask=attention_mask, ss=ss, os=os, se=se, oe=oe)
        # Encode S:64*2304
        S = self.PCNN(Xp, Xe, X_Mask) #PCNN
        U = self.SAN(Xp, Xe) #Attention  U:64*2304
        # gate
        X = self.selective_gate(S, U)#X:64*2304
        # X = torch.cat((S,U), dim=-1)
        # input_ids = torch.as_tensor(input_ids, dtype=torch.float).to(0)
        # len = torch.IntTensor([64, 180])
        Xp_shape_0 = Xp.shape[0]
        len = torch.full([Xp_shape_0],180)
        # len = torch.full([64],180)
        # text_out, (_, _) = self.text_lstm(input_ids, len)#torch.Size([16, 26, 600])
        # text_out：64*180*4608
        # print("Xp",Xp.shape)
        text_out, (_, _) = self.text_lstm(pooled_output, len)#torch.Size([16, 26, 600])
        # print("text_out",text_out.shape)
        return E, T, X,pooled_output
        # return E, T, X,text_out

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
        pooled_output = outputs[0]#pooled_output([64, 180, 768])
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
        return E, T, Xp, Xe,pooled_output

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
