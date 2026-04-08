import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self, z_dim):
        super().__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 84 → 42
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),  # 42 → 21
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),  # 21 → 10
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 1, 1),  # 10 → 10
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),  # 10 → 10
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),                  # 256 * 10 * 10 = 25600
            nn.Linear(25600, z_dim),
            nn.LayerNorm(z_dim)
        )

        # DECODER
        self.decoder_fc = nn.Linear(z_dim, 25600)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 10 → 20
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 20 → 40
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 40 → 80
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 80 → 160
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)

        x_hat = self.decoder_fc(z)
        x_hat = x_hat.view(-1, 256, 10, 10)

        x_hat = self.decoder(x_hat)

        # crop back to 84x84
        x_hat = x_hat[:, :, :84, :84]

        return x_hat, z

    def decode(self, z):
        x_hat = self.decoder_fc(z)
        x_hat = x_hat.view(-1, 256, 10, 10)
        x_hat = self.decoder(x_hat)
        x_hat = x_hat[:, :, :84, :84]
        return x_hat