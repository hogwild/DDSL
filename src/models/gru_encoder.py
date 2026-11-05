from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoderClassifier(nn.Module):
    """
    K/Q/G/M token sequence -> joint_state classification via a GRU encoder.

    This model is an *architecture-independent learnability probe*, not a
    replacement for TransformerEncoderClassifier. It intentionally shares:

        - the same input contract: forward(K, Q, G, M) -> logits [B, num_classes]
        - the same K/Q/G/M embedding-and-concat front end
        - the same training/evaluation code path (train_one_epoch, evaluate),
          since both only ever call model(K, Q, G, M)

    and intentionally differs only in using a recurrent (GRU) encoder instead
    of a self-attention encoder, so that its macro-F1 trajectory across DDSL
    iterations can be compared against the Transformer's macro-F1 trajectory
    on the *same* labels, as evidence for or against the claim that rising
    learnability across iterations is a property of the labels themselves
    rather than an artifact of the Transformer fitting to logits it produced
    itself in a prior iteration.

    IMPORTANT (structural safety, not just documentation):
    This model's logits must NEVER be exported via run_export_transformer_logits.py or
    fed into the Viterbi energy function. It is evaluation-only. There is
    deliberately no run_export_logits_gru.py in this codebase -- only
    TransformerEncoderClassifier participates in the DDSL label-feedback
    loop. Do not add such a script without re-reading this docstring.
    """

    def __init__(
        self,
        num_K: int,
        num_Q: int,
        num_G: int,
        num_M: int,
        num_classes: int = 9,
        d_model: int = 64,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling: str = "last",  # "last" or "mean"
    ):
        super().__init__()

        if pooling not in {"last", "mean"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.pooling = pooling
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.emb_K = nn.Embedding(num_K, d_model)
        self.emb_Q = nn.Embedding(num_Q, d_model)
        self.emb_G = nn.Embedding(num_G, d_model)
        self.emb_M = nn.Embedding(num_M, d_model)

        self.input_proj = nn.Linear(d_model * 4, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)

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
            True = valid token, False = padding.
            The current KQGMJointDataset produces fixed-length seq_len
            windows with no padding, so this is normally None; it is
            supported anyway so the model degrades gracefully if that
            ever changes.
        """
        B, T = K.shape

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
        x = self.input_dropout(x)

        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1).clamp(min=1)
            if bool((lengths > T).any()):
                raise ValueError("attention_mask implies lengths exceeding sequence length.")
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            packed_out, h_n = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T,
            )
        else:
            lengths = torch.full((B,), T, dtype=torch.long, device=K.device)
            out, h_n = self.gru(x)

        if self.pooling == "last":
            if self.bidirectional:
                # h_n: [num_layers * 2, B, hidden_size].
                # Layout is (layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...),
                # so the last layer's forward/backward final states are -2/-1.
                # These already respect per-sample lengths via packing above.
                fwd = h_n[-2]
                bwd = h_n[-1]
                pooled = torch.cat([fwd, bwd], dim=-1)
            else:
                idx = lengths - 1
                pooled = out[torch.arange(B, device=out.device), idx]
        else:  # mean pooling
            mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.float().unsqueeze(-1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled)

        return logits