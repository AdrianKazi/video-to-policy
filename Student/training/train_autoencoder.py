import os
import torch
import matplotlib.pyplot as plt

from utils.losses import weighted_mse
from utils.save_model import save_model
from evaluation.autoencoder.orig_recon_comparison import autoencoder_reconstruction


def train_autoencoder(train_dataset, AutoEncoder):

    model = AutoEncoder(z_dim=64)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    losses = []

    print('\n[TRAIN AE]\n')

    for epoch in range(40):
        total_loss = 0

        for seq in train_dataset:
            x = seq

            x_hat, _ = model(x)

            loss = weighted_mse(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_dataset)
        losses.append(epoch_loss)

        print(f"[AE] Epoch {epoch:>2} | Loss {epoch_loss:.6f}")

    run_dir = save_model(
        model,
        losses,
        model_name="autoencoder"
    )

    plt.figure()
    plt.plot(losses)
    plt.title('AutoEncoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    plot_path = os.path.join(run_dir, 'loss.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"[PLOT SAVED] {plot_path}")

    autoencoder_reconstruction(model, train_dataset, run_dir)

    print('\n[AE TRAINING DONE]\n')

    return model, losses