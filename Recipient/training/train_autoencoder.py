import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =========================
# LOSS
# =========================
def weighted_mse(x_hat, x):
    return F.mse_loss(x_hat, x)


# =========================
# SAVE MODEL
# =========================
def save_model(model, losses, model_type, model_name="model"):
    run_dir = os.path.join("runs", f"{model_type}_{model_name}")
    os.makedirs(run_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    torch.save(losses, os.path.join(run_dir, "losses.pt"))

    return run_dir


# =========================
# TRAIN
# =========================
def train_autoencoder(train_dataset, AutoEncoder):

    model = AutoEncoder(z_dim=64)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    losses = []

    print('\n[TRAIN AE]\n')

    for epoch in range(50):
        total_loss = 0

        for seq in train_dataset:
            # seq: (T,1,84,84)
            x = seq  # traktujemy jako batch

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
        'autoencoder',
        model_name="ae_z64"
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
    print('\n[AE TRAINING DONE]\n')

    return model, losses