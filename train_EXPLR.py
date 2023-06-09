import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import MathProcessor
from evaluation import get_f1, get_macro_f1
from model import REModel
from torch.cuda.amp import GradScaler
import wandb
import time
#         add WxPusher
import requests
import json

# wxpusher
headers = {'content-type': "application/json"}
body = {
  "appToken":"AT_ovoKNeFYClICWUKHnbF3BhYLkflxjy77",
  "content":"acc=",
  "summary":"BERT_PAG",
  "contentType":1,
  "topicIds":[],
  "uids":["UID_YicrCnEFRs6teHjQXis8EQ9nVoY3"]
}

def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size,
                                  shuffle=True, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader) *
                      args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)  # update every epoch
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'os': batch[4].to(args.device),
                      'se': batch[5].to(args.device),
                      'oe': batch[6].to(args.device),
                      'pos1': batch[7].to(args.device),
                      'pos2': batch[8].to(args.device),
                      'mask': batch[9].to(args.device),
                      'matrix': batch[10].to(args.device)
                      }
            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                ExpLR.step()
                model.zero_grad()
                wandb.log({'loss': loss.item()}, step=num_steps)
                print('loss', loss.item())

            if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                for tag, features in benchmarks:
                    f1, output = evaluate(args, model, features, tag=tag)
                    wandb.log(output, step=num_steps)

    for tag, features in benchmarks:
        f1, output = evaluate(args, model, features, tag=tag)
        wandb.log(output, step=num_steps)
        body['content'] = str(output)+" F1: "+str(f1)
        # send WxPusher
        ret = requests.post('http://wxpusher.zjiecode.com/api/send/message', data=json.dumps(body), headers=headers)


def evaluate(args, model, features, tag='dev'):
    dataloader = DataLoader(
        features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  'se': batch[5].to(args.device),
                  'oe': batch[6].to(args.device),
                  'pos1': batch[7].to(args.device),
                  'pos2': batch[8].to(args.device),
                  'mask': batch[9].to(args.device)
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    p, r, max_f1, macro_p, macro_r, macro_f1 = get_f1(args,keys, preds)

    output = {
        tag + "_micro_p": p * 100,
        tag + "_micro_r": r * 100,
        tag + "_micro_f1": max_f1 * 100,
        tag + "_macro_p": macro_p * 100,
        tag + "_macro_r": macro_r * 100,
        tag + "_macro_f1": macro_f1 * 100,
    }
    print(output)
    # return max_f1, output  # max_f1是micro_f1
    return macro_f1, output


def main():
    parser = argparse.ArgumentParser()

    dataset = "./dataset/literature"
    # dataset = "./dataset/FinRE"
    # parser.add_argument("--data_dir", default="./dataset/literature", type=str)
    parser.add_argument("--data_dir", default=dataset, type=str)
    if "literature" in dataset:
        parser.add_argument("--num_class", type=int, default=10)
    elif "FinRE" in dataset:
        parser.add_argument("--num_class", type=int, default=44)

    # parser.add_argument("--num_class", type=int, default=10)
    # parser.add_argument("--num_class", type=int, default=44)
    parser.add_argument("--model_name_or_path",
                        default="bert-base-chinese", type=str)
    # parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
    #                     help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")
    parser.add_argument("--input_format", default="typed_entity_marker", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=180, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=64, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--evaluation_steps", type=int, default=100,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="BERTPAG")
    parser.add_argument("--run_name", type=str, default="re-tacred")

    args = parser.parse_args()
    wandb.init(project=args.project_name, name=args.run_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    model = REModel(args, config)
    model.to(0)

    train_file = os.path.join(args.data_dir, "train2.json")
    dev_file = os.path.join(args.data_dir, "dev2.json")
    test_file = os.path.join(args.data_dir, "test2.json")

    # train_file = os.path.join(args.data_dir, "dev2.json")
    # dev_file = os.path.join(args.data_dir, "dev2.json")
    # test_file = os.path.join(args.data_dir, "dev2.json")

    # train_file = os.path.join(args.data_dir, "train.json")
    # dev_file = os.path.join(args.data_dir, "dev.json")
    # test_file = os.path.join(args.data_dir, "test.json")


    processor = MathProcessor(args, tokenizer)
    train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
    )
    since = time.time()
    train(args, model, train_features, benchmarks)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
    main()
