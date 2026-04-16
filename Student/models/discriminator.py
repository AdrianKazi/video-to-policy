import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, z_dim, hidden_dim=256):
        super().__init__()

        # Concat of (z_t, z_{t+1}) → 2 * z_dim
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z_t, z_t1):
        x = torch.cat([z_t, z_t1], dim=-1)
        return self.net(x)

    def reward(self, z_t, z_t1):
        with torch.no_grad():
            d = self.forward(z_t, z_t1)
            # Clamp for stabiloty
            return -torch.log(d + 1e-8)
