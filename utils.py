import torch
import random
import numpy as np
import constant


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    max_len = constant.MAX_LENGTH
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    se = [f["se"] for f in batch]
    oe = [f["se"] for f in batch]
    pos1 = [f["pos1"] for f in batch]
    pos2 = [f["pos2"] for f in batch]
    mask = [f["mask"] for f in batch]
    matrix = [f["matrix"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    pos1 = torch.tensor(pos1, dtype=torch.long)
    pos2 = torch.tensor(pos2, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    se = torch.tensor(se, dtype=torch.long)
    oe = torch.tensor(oe, dtype=torch.long)
    matrix = torch.tensor(matrix, dtype=torch.float)

    # 新增matrix
    output = (input_ids, input_mask, labels, ss, os, se, oe, pos1, pos2, mask,matrix)
    return output
