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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--glove_path", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)  # number of articles per batch
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build dataset and vocab
    dataset = SummDataset(args.train_json, build_vocab=True)
    vocab = dataset.vocab

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # model
    model = ExtractiveSummarizer(vocab, embed_dim=300, hidden_size=256, glove_path=args.glove_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch} loss: {loss:.4f}")
        # save checkpoint
        save_model(model, args.save_path)
        print("Saved model to", args.save_path)

if __name__ == "__main__":
    main()
