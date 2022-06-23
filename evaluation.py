import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report

def get_f1(key, prediction):
    correct_by_relation = ((key == prediction) & (prediction != 0)).astype(np.int32).sum()
    guessed_by_relation = (prediction != 0).astype(np.int32).sum()
    gold_by_relation = (key != 0).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    with open('./dataset/rels.txt','r') as fp:
        labels = fp.read().strip().split('\n')  
    macro_p, macro_r, macro_f1 = get_macro_f1(key, prediction, labels)
    return prec_micro, recall_micro, f1_micro, macro_p, macro_r, macro_f1

def get_macro_f1(key, predicition, labels):
    p = precision_score(key, predicition, average="macro")
    r = recall_score(key, predicition, average="macro")
    macro_f1 = f1_score(key, predicition, average="macro")
    report = classification_report(key, predicition, target_names=labels, digits=5)
    print("Macro-", "f1: ", macro_f1, " recall: ", r, " precision: ", p)
    print("report======", report)
    return p, r, macro_f1
    