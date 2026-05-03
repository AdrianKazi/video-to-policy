
import torch
import torch.nn as nn


class IDM(nn.Module):
    """Predicts a_t from (z_t, z_{t+1})."""

    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_t1], dim=-1))


class IDMContext(nn.Module):
    """Predicts action from a window of latents."""

    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int = 256,
                 n_heads: int = 4, num_layers: int = 2, max_seq_len: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(z_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = z_seq.shape
        device = z_seq.device
        x = self.input_proj(z_seq) + self.pos_embed(torch.arange(T, device=device).unsqueeze(0))
        x = self.transformer(x)
        return self.head(x[:, -1])
