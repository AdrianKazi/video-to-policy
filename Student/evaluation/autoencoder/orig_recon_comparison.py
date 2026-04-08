import os
import torch
import matplotlib.pyplot as plt


def autoencoder_reconstruction(model, dataset, run_dir):

    os.makedirs(run_dir, exist_ok=True)

    model.eval()

    x = dataset[0]

    with torch.no_grad():
        x_hat, _ = model(x)

    num_rows = min(20, x.shape[0])

    fig, axs = plt.subplots(num_rows, 2, figsize=(6, 2*num_rows))

    for i in range(num_rows):
        axs[i, 0].imshow(x[i].squeeze(), cmap='gray')
        axs[i, 0].axis('off')
        if i == 0:
            axs[i, 0].set_title("orig")

        axs[i, 1].imshow(x_hat[i].squeeze(), cmap='gray')
        axs[i, 1].axis('off')
        if i == 0:
            axs[i, 1].set_title("recon")

    plt.tight_layout()

    save_path = os.path.join(run_dir, "reconstruction.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[RECON SAVED] {save_path}")