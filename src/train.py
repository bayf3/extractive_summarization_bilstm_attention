# src/train.py
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from utils import save_model

import numpy as np
from inference import predict_summary
from evaluate import rouge_1, rouge_l


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for batch in tqdm(dataloader, desc="Train"):
        ids = batch["ids"]
        word_ids_list = batch["word_ids"]
        lengths_list = batch["lengths"]
        labels_list = batch["labels"]

        optimizer.zero_grad()
        batch_loss = 0.0
        count = 0
        for word_ids, lengths, labels in zip(word_ids_list, lengths_list, labels_list):
            # move to device
            word_ids = word_ids.to(device)          # [num_sent, max_len]
            lengths = lengths.to(device)
            labels = labels.to(device)

            if word_ids.numel() == 0:
                continue

            logits, attn = model(word_ids, lengths)  # [num_sent], [num_sent]
            loss = criterion(logits, labels)
            batch_loss += loss
            count += 1

        if count == 0:
            continue
        batch_loss = batch_loss / count
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
    return total_loss

@torch.no_grad()
def eval_epoch(model, dataloader, device, strategy="topk"):
    model.eval()

    r1_f, rl_f = [], []

    for batch in tqdm(dataloader, desc="Val"):
        ids_list = batch["word_ids"]
        lens_list = batch["lengths"]
        raw_sents_batch = batch["raw_sents"]
        refs_batch = batch["highlights"]

        for i in range(len(ids_list)):
            word_ids = ids_list[i].to(device)
            lengths = lens_list[i].to(device)

            if word_ids.numel() == 0:
                continue

            # forward
            output = model(word_ids, lengths)
            if len(output) == 3:
                logits, attn, vectors = output
            else:
                logits, attn = output
                vectors = None

            sent_scores = torch.sigmoid(logits)

            # generate summary
            pred_sents = predict_summary(
                article_sents=raw_sents_batch[i],
                sent_scores=sent_scores,
                sent_vectors=vectors,
                strategy=strategy
            )

            ref_sents = refs_batch[i]

            r1 = rouge_1(pred_sents, ref_sents)
            rl = rouge_l(pred_sents, ref_sents)

            r1_f.append(r1["f"])
            rl_f.append(rl["f"])

    return {
        "r1_f": float(np.mean(r1_f)) if r1_f else 0.0,
        "rl_f": float(np.mean(rl_f)) if rl_f else 0.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--glove_path", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)  # number of articles per batch
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--val_strategy", default="topk") # strategy: 'topk', 'dynamic', 'mmr', 'dynamic_mmr'

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build dataset and vocab
    # dataset = SummDataset(args.train_json, build_vocab=True, save_vocab_path="./vocab/vocab.pkl")
    # vocab = dataset.vocab
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # train dataset (build vocab)
    train_dataset = SummDataset(
        args.train_json,
        build_vocab=True,
        save_vocab_path="./vocab/vocab.pkl"
    )

    # val dataset (load vocab!)
    val_dataset = SummDataset(
        args.val_json,
        vocab=train_dataset.vocab,
        build_vocab=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,          # 验证建议 batch=1，逻辑最清晰
        shuffle=False,
        collate_fn=collate_fn
    )


    # model
    model = ExtractiveSummarizer(train_dataset.vocab, embed_dim=300, hidden_size=256, glove_path=args.glove_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    val_scores = eval_epoch(
        model, val_loader, device, strategy=args.val_strategy
    )
    print(
        f"ROUGE-1 F1: {val_scores['r1_f']:.4f} | "
        f"ROUGE-L F1: {val_scores['rl_f']:.4f}"
    )

    best_rl = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        print(f"Epoch {epoch} loss: {train_loss:.4f}")

        val_scores = eval_epoch(
            model, val_loader, device, strategy=args.val_strategy
        )
        print(
            f"ROUGE-1 F1: {val_scores['r1_f']:.4f} | "
            f"ROUGE-L F1: {val_scores['rl_f']:.4f}"
        )

        # save best model
        if val_scores["rl_f"] > best_rl:
            best_rl = val_scores["rl_f"]
            save_model(model, args.save_path)
            print(f"Model saved (ROUGE-L={best_rl:.4f})")
    
    # save_model(model, args.save_path)

if __name__ == "__main__":
    main()
