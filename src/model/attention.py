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
        # self.query = nn.Parameter(torch.empty(hidden_dim))
        # nn.init.xavier_uniform_(self.query.data)

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
        # weights = F.softmax(scores, dim=0)
        return scores  # 返回 logits 以供二分类使用


class MultiHeadAttention(nn.Module):
    """
    [新版/进阶] 多头注意力 (Multi-Head Attention)
    使用多头机制增强特征捕捉能力。
    """

    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        # 确保整除
        assert hidden_dim % num_heads == 0, f"Hidden dim {hidden_dim} must be divisible by {num_heads}"

        self.hidden_dim = hidden_dim

        # PyTorch 原生 MultiheadAttention
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         batch_first=True)  # 输入形状为 [Batch, Seq, Dim]

        # 最终打分层
        self.score_layer = nn.Linear(hidden_dim, 1)

    def forward(self, sent_vecs):
        """
        Args:
            sent_vecs: [N_sent, hidden_dim]
        Returns:
            scores: [N_sent] (Logits, 未归一化)
        """
        if sent_vecs.size(0) == 0:
            return torch.tensor([], device=sent_vecs.device)

        # 1. 计算文档向量 (Query)
        # 这里简单使用均值，MHA内部也会处理特征交互
        doc_vec = torch.mean(sent_vecs, dim=0)

        # 2. 构造输入 (Batch=1)
        # Query: 句子 [1, N_sent, Dim]
        query = sent_vecs.unsqueeze(0)
        # Key/Value: 文档中心 [1, 1, Dim]
        key = doc_vec.unsqueeze(0).unsqueeze(0)
        value = key

        # 3. 注意力计算
        # attn_output: [1, N_sent, Dim]
        attn_output, _ = self.mha(query, key, value)

        # 4. 残差连接 + 降维
        enhanced_sents = attn_output.squeeze(0) + sent_vecs

        # 5. 映射为分数
        scores = self.score_layer(enhanced_sents).squeeze(-1)

        return scores