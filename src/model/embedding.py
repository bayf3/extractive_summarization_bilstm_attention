# src/model/embedding.py
import os
import numpy as np
import torch
import torch.nn as nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class GloveEmbedding(nn.Module):
    """
    GloVe Embedding layer.
    - vocab: mapping word->idx
    - embedding matrix: initialized from glove_path if provided, else random.
    """
    def __init__(self, vocab, embedding_dim=300, glove_path=None, trainable=True):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)

        # Initialize weights
        self.init_weights(glove_path)
        self.embedding.weight.requires_grad = trainable

    def init_weights(self, glove_path):
        # default random normal for all
        std = 0.01
        weight = np.random.normal(scale=std, size=(self.vocab_size, self.embedding_dim)).astype(np.float32)
        # pad vector zero
        weight[0] = np.zeros(self.embedding_dim, dtype=np.float32)

        if glove_path is not None and os.path.exists(glove_path):
            print("Loading GloVe from", glove_path)
            # load glove into dict but only for words in vocab
            glove = {}
            with open(glove_path, 'r', encoding='utf8', errors='ignore') as f:
                for line in f:
                    parts = line.rstrip().split(' ')
                    token = parts[0]
                    vec = parts[1:]
                    if len(vec) != self.embedding_dim:
                        continue
                    glove[token] = np.asarray(vec, dtype=np.float32)
            matched = 0
            for w, idx in self.vocab.items():
                if w in glove:
                    weight[idx] = glove[w]
                    matched += 1
            print(f"Matched {matched}/{len(self.vocab)} words in GloVe.")
        else:
            if glove_path is not None:
                print("Warning: glove_path provided but file not found:", glove_path)

        self.embedding.weight.data.copy_(torch.from_numpy(weight))

    def forward(self, input_ids):
        """
        input_ids: LongTensor [..., seq_len]
        returns: [..., seq_len, embedding_dim]
        """
        return self.embedding(input_ids)
