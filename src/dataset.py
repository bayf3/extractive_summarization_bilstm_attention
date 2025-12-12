# src/dataset.py
import json
import os
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from collections import Counter
import random

try:
    # prefer nltk tokenizer for english
    import nltk
    nltk.data.find('tokenizers/punkt')
    from nltk.tokenize import word_tokenize
except Exception:
    # fallback to simple split
    def word_tokenize(x):
        return x.split()

PAD = "<pad>"
UNK = "<unk>"

class SummDataset(Dataset):
    """
    Dataset expects JSON list where each element is dict:
      { "id": "...", "sentences": [...], "labels": [...], "highlights": [...] }
    Tokenization is performed on the fly during __getitem__ or in collate.
    """
    def __init__(self, json_path, max_sent_len=100, min_freq=1, build_vocab=True, vocab=None):
        self.data = self._load(json_path)
        self.max_sent_len = max_sent_len
        if build_vocab:
            self.vocab = self.build_vocab(self.data, min_freq=min_freq)
        else:
            if vocab is None:
                raise ValueError("vocab required if build_vocab=False")
            self.vocab = vocab

    def _load(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    def build_vocab(self, data, min_freq=1):
        counter = Counter()
        for item in data:
            for s in item.get("sentences", []):
                toks = word_tokenize(s.lower())
                counter.update(toks)
        # special tokens
        vocab = {PAD:0, UNK:1}
        idx = 2
        for tok, freq in counter.most_common():
            if freq < min_freq:
                break
            if tok in vocab:
                continue
            vocab[tok] = idx
            idx += 1
        print(f"Built vocab size: {len(vocab)}")
        return vocab

    def __len__(self):
        return len(self.data)

    def sentence_to_ids(self, sentence):
        toks = word_tokenize(sentence.lower())
        toks = toks[:self.max_sent_len]
        ids = [self.vocab.get(t, self.vocab.get(UNK)) for t in toks]
        return ids, len(ids)

    def __getitem__(self, idx):
        item = self.data[idx]
        sents = item.get("sentences", [])
        labels = item.get("labels", [])
        sent_ids = []
        sent_lens = []
        for s in sents:
            ids, l = self.sentence_to_ids(s)
            sent_ids.append(ids)
            sent_lens.append(l)
        return {
            "id": item.get("id"),
            "sentences": sents,
            "sent_ids": sent_ids,
            "sent_lens": sent_lens,
            "labels": labels,
            "highlights": item.get("highlights", [])
        }

def collate_fn(batch, pad_idx=0):
    """
    batch: list of dataset items (one article per element).
    We return a list-based batch:
      {
        "ids": [...],
        "word_ids": [ LongTensor(num_sent, max_len) for each article ],
        "lengths": [ LongTensor(num_sent) for each article ],
        "labels": [ FloatTensor(num_sent) for each article ],
        "highlights": [...]
      }
    """
    ids = []
    word_id_tensors = []
    length_tensors = []
    label_tensors = []
    highlights = []
    for item in batch:
        ids.append(item["id"])
        sent_ids = item["sent_ids"]
        sent_lens = item["sent_lens"]
        if len(sent_ids) == 0:
            # empty article
            word_id_tensors.append(torch.zeros((0,0), dtype=torch.long))
            length_tensors.append(torch.zeros((0,), dtype=torch.long))
            label_tensors.append(torch.zeros((0,), dtype=torch.float))
            highlights.append(item.get("highlights", []))
            continue
        max_len = max(len(s) for s in sent_ids)
        padded = []
        for s in sent_ids:
            padded.append(s + [pad_idx] * (max_len - len(s)))
        word_id_tensors.append(torch.tensor(padded, dtype=torch.long))  # [num_sent, max_len]
        length_tensors.append(torch.tensor(sent_lens, dtype=torch.long))
        label_tensors.append(torch.tensor(item.get("labels", [0]*len(sent_ids)), dtype=torch.float))
        highlights.append(item.get("highlights", []))

    return {
        "ids": ids,
        "word_ids": word_id_tensors,
        "lengths": length_tensors,
        "labels": label_tensors,
        "highlights": highlights
    }
