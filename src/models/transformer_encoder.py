from __future__ import annotations

import torch
import torch.nn as nn
import math 


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (works well and no extra params)."""
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class TransformerEncoderClassifier(nn.Module):
    """
    K/Q/G/M/W token sequence -> joint_state classification.

    Input:
        batch["K"], batch["Q"], batch["G"], batch["M"]
        shape: [B, T]

    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        num_K: int,
        num_Q: int,
        num_G: int,
        num_M: int,
        num_classes: int = 9,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        max_seq_len: int = 256,
        pooling: str = "last",  # "last", "mean", or "cls"
    ):
        super().__init__()

        if pooling not in {"last", "mean", "cls"}:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.pooling = pooling
        self.d_model = d_model

        self.emb_K = nn.Embedding(num_K, d_model)
        self.emb_Q = nn.Embedding(num_Q, d_model)
        self.emb_G = nn.Embedding(num_G, d_model)
        self.emb_M = nn.Embedding(num_M, d_model)

        self.input_proj = nn.Linear(d_model * 4, d_model)

        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)
        # self.pos_emb = PositionalEncoding(
        #                 d_model=d_model,
        #                 max_len=max_seq_len + 1,
        #                 dropout=dropout,
        #             )

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        K: torch.Tensor,
        Q: torch.Tensor,
        G: torch.Tensor,
        M: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        K/Q/G/M:
            LongTensor [B, T]

        attention_mask:
            Optional BoolTensor [B, T]
            True = valid token
            False = padding
        """

        B, T = K.shape
        # x = self.emb_K(K) + self.emb_Q(Q) + self.emb_G(G) + self.emb_M(M)
        x = torch.cat(
            [
                self.emb_K(K),
                self.emb_Q(Q),
                self.emb_G(G),
                self.emb_M(M),
            ],
            dim=-1,
        )

        x = self.input_proj(x)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, 1, self.d_model)
            x = torch.cat([cls, x], dim=1)

            if attention_mask is not None:
                cls_mask = torch.ones(
                    B, 1,
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        seq_len = x.size(1)
        # x = self.pos_emb(x)
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len "
                f"{self.pos_emb.num_embeddings - 1}"
            )

        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_ids)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        h = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )

        if self.pooling == "cls":
            pooled = h[:, 0]

        elif self.pooling == "last":
            if attention_mask is None:
                pooled = h[:, -1]
            else:
                lengths = attention_mask.long().sum(dim=1).clamp(min=1)
                idx = lengths - 1
                pooled = h[torch.arange(B, device=h.device), idx]

        else:  # mean pooling
            if attention_mask is None:
                pooled = h.mean(dim=1)
            else:
                mask = attention_mask.float().unsqueeze(-1)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled)

        return logits