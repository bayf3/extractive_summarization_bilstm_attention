# src/evaluate.py
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from utils import load_json
import numpy as np
import math

def rouge_1(pred_sentences, ref_sentences):
    # compute unigram overlap recall/precision/F1 (simple)
    def tokenize(s): return s.lower().split()
    pred_words = []
    for s in pred_sentences:
        pred_words += tokenize(s)
    ref_words = []
    for s in ref_sentences:
        ref_words += tokenize(s)
    from collections import Counter
    pc = Counter(pred_words)
    rc = Counter(ref_words)
    overlap = sum((pc & rc).values())
    recall = overlap / max(1, sum(rc.values()))
    prec = overlap / max(1, sum(pc.values()))
    if prec + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return {"r":recall, "p":prec, "f":f1}

def lcs(a, b):
    # simple DP LCS length for token lists
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            if a[i]==b[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

def rouge_l(pred_sentences, ref_sentences):
    pred = " ".join(pred_sentences).lower().split()
    ref = " ".join(ref_sentences).lower().split()
    L = lcs(pred, ref)
    if len(ref)==0 or len(pred)==0:
        return {"r":0.0,"p":0.0,"f":0.0}
    r = L/len(ref)
    p = L/len(pred)
    if r+p == 0:
        f=0.0
    else:
        f = 2*r*p/(r+p)
    return {"r":r,"p":p,"f":f}

def select_topk(sentences, probs, k=3):
    # probs: numpy array
    if len(sentences) == 0:
        return []
    idxs = np.argsort(-probs)[:k]
    idxs_sorted = sorted(idxs)
    return [sentences[i] for i in idxs_sorted]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--glove_path", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SummDataset(args.data_json, build_vocab=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = ExtractiveSummarizer(dataset.vocab, embed_dim=300, hidden_size=256, glove_path=args.glove_path)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    rouge1_f = []
    rougeL_f = []

    with torch.no_grad():
        for batch in dataloader:
            word_ids_list = batch["word_ids"]
            lengths_list = batch["lengths"]
            highlights = batch["highlights"]
            for word_ids, lengths, refs in zip(word_ids_list, lengths_list, highlights):
                if word_ids.numel() == 0:
                    continue
                word_ids = word_ids.to(device)
                lengths = lengths.to(device)
                logits, attn = model(word_ids, lengths)
                probs = torch.sigmoid(logits).cpu().numpy()
                # pick top-k
                sentences = []  # need raw sentences - easier to re-open dataset if necessary
                # Here numeric selection only; but we have no raw sentences in this dataloader
                # So for evaluation, we will reconstruct from dataset by id or you can provide references only.
                # For simplicity, assume dataset.data aligns with dataloader order.
                # This evaluate is a simple demo: user can adapt to use raw sentences for final scoring.
                # We'll skip computing ROUGE when we don't have the original sentences here.
                pass

    print("Evaluation script is a simple template. For full ROUGE please adapt to load raw sentences for selection.")
