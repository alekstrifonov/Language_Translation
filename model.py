#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Used code from the pytorch documentation"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class LanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        startTokenIdx,
        endTokenIdx,
        transTokenIdx,
        padTokenIdx,
        layers,
        embed_size,
        hidden_size,
        n_head,
        dropout,
    ):
        super(LanguageModel, self).__init__()
        self.startTokenIdx = startTokenIdx
        self.endTokenIdx = endTokenIdx
        self.padTokenIdx = padTokenIdx
        self.transTokenIdx = transTokenIdx
        self.n_head = n_head

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.layers = layers

        self.embedding = nn.Embedding(
            self.vocab_size, self.embed_size, padding_idx=self.padTokenIdx
        )
        self.positional_encoding = PositionalEncoding(self.embed_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=self.n_head,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)

        self.out_projection = nn.Linear(self.embed_size, self.vocab_size)

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents_padded = [s + (m - len(s)) * [self.padTokenIdx] for s in source]
        return torch.tensor(
            sents_padded, dtype=torch.long, device=device
        )  # shape=(batch_size, seq_len)

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    def forward(self, source):
        X = self.preparePaddedBatch(source)  # (batch_size, seq_len)
        padding_mask = X == self.padTokenIdx  # (batch_size, seq_len)

        seq_len = X.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=X.device) * float("-inf"), diagonal=1
        )
        embedded = self.positional_encoding(
            self.embedding(X)
        )  # (batch_size, seq_len, embed_size)

        decoder = self.encoder(
            embedded, mask=causal_mask, src_key_padding_mask=padding_mask
        ) # (batch_size, ..., embed_size)

        Z = self.out_projection(decoder)[:, :-1, :].reshape(-1, self.vocab_size)
        Y_bar = X[:, 1:].reshape(-1)
        H = torch.nn.functional.cross_entropy(
            Z, Y_bar, ignore_index=self.padTokenIdx)

        return H

    def generate(self, prefix, limit=1000):
        return result
