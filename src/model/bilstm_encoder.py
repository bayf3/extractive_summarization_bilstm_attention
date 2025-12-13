# src/model/bilstm_encoder.py
import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    """
    Encode a batch of sentences (padded) into sentence vectors via BiLSTM.
    Input:
        x: [batch_sentences, max_len, embed_dim]
        lengths: list or tensor of lengths (batch_sentences,)
    Output:
        sent_vecs: [batch_sentences, hidden_dim*2]  (we use last hidden states)
    """
    def __init__(self, embed_dim=300, hidden_size=256, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """
        x: FloatTensor [B, L, E]
        lengths: LongTensor [B]
        """
        x = self.input_dropout(x)
        # pack
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # h_n: [num_layers*2, B, hidden]
        # take last layer's forward and backward:
        # since num_layers=1 -> indices 0 and 1
        forward_h = h_n[0]  # [B, hidden]
        backward_h = h_n[1]  # [B, hidden]
        sent_vec = torch.cat([forward_h, backward_h], dim=-1)  # [B, hidden*2]
        sent_vec = self.output_dropout(sent_vec)
        return sent_vec
