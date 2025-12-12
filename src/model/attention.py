# src/model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """
    Additive attention (Bahdanau style) with learnable query vector.
    Computes attention over sentence vectors per article.
    """

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        # query vector parameter
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, sent_vecs):
        """
        sent_vecs: [N_sent, hidden_dim]
        returns:
            attn_weights: [N_sent] (softmax over N_sent)
        """
        if sent_vecs.size(0) == 0:
            return torch.tensor([], device=sent_vecs.device)

        # expand query
        q = self.query.unsqueeze(0).expand(sent_vecs.size(0), -1)  # [N, H]
        u = torch.tanh(self.W(sent_vecs) + self.U(q))
        scores = self.v(u).squeeze(-1)  # [N]
        weights = F.softmax(scores, dim=0)
        return weights
