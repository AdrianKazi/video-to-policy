"""Sequential latent predictor."""

import torch
import torch.nn as nn


class LatentLSTM(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=z_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                z_seq,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(z_seq)
        return self.head(h_n[-1])


class LatentTransformer(nn.Module):
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(z_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = z_seq.shape
        device = z_seq.device

        x = self.input_proj(z_seq)
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device, dtype=x.dtype)

        padding_mask = None
        if lengths is not None:
            seq_range = torch.arange(T, device=device).unsqueeze(0)
            bool_mask = seq_range >= lengths.unsqueeze(1)
            padding_mask = torch.zeros_like(bool_mask, dtype=x.dtype).masked_fill(bool_mask, float("-inf"))

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        if lengths is not None:
            last_idx = (lengths - 1).clamp(min=0).long()
        else:
            last_idx = torch.full((B,), T - 1, dtype=torch.long, device=device)

        h_last = x[torch.arange(B, device=device), last_idx]
        return self.head(h_last)


def build_model(model_type: str, **kwargs) -> nn.Module:
    if model_type == "transformer":
        return LatentTransformer(**kwargs)
    return LatentLSTM(**kwargs)
