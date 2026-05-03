
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Autoencoder.device_util import pick_device
from Autoencoder.network import AutoEncoder
from IDM.config import IDMConfig
from IDM.dataset import PairDataset, ContextDataset, load_and_encode
from IDM.metrics import compute_metrics
from IDM.network import IDM, IDMContext


def _latest_ae(cfg: IDMConfig) -> Path:
    runs = sorted(cfg.ae_runs_dir.glob("autoencoder_*"))
    for d in reversed(runs):
        if (d / "model.pth").is_file():
            return d / "model.pth"
    raise FileNotFoundError(f"No AE checkpoint under {cfg.ae_runs_dir}")


def run_train(cfg: IDMConfig, ae_checkpoint: Path | None = None) -> tuple[Path, dict]:
    device = pick_device(cfg.device)

    if ae_checkpoint is None:
        ae_checkpoint = _latest_ae(cfg)

    ae = AutoEncoder(z_dim=cfg.z_dim).to(device)
    ae.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    data_path = cfg.data_dir / "labeled_rollouts.pt"
    train_z, train_a, test_z, test_a = load_and_encode(
        data_path, ae, device, cfg.test_ratio, cfg.seed)

    if cfg.model == "context":
        train_ds = ContextDataset(train_z, train_a, cfg.context_len)
        test_ds = ContextDataset(test_z, test_a, cfg.context_len)
        model = IDMContext(
            cfg.z_dim, cfg.action_dim, cfg.hidden_dim,
            cfg.n_heads, cfg.num_layers, cfg.context_len + 1,
        ).to(device)
    else:
        train_ds = PairDataset(train_z, train_a)
        test_ds = PairDataset(test_z, test_a)
        model = IDM(cfg.z_dim, cfg.action_dim, cfg.hidden_dim).to(device)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=cfg.batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=cfg.batch_size)
    print(f"[IDM] {cfg.model}, {sum(p.numel() for p in model.parameters()):,} params, "
          f"train {len(train_ds)}, test {len(test_ds)}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_losses, test_losses = [], []
    t0 = time.time()

    stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_dir = cfg.runs_dir / f"idm_{cfg.model}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, n_b = 0.0, 0
        for batch in train_dl:
            if cfg.model == "context":
                z_win, a_true = batch[0].to(device), batch[1].to(device)
                a_pred = model(z_win)
            else:
                zt, a_true, zt1 = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                a_pred = model(zt, zt1)

            loss = F.mse_loss(a_pred, a_true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += loss.item()
            n_b += 1

        model.eval()
        te_loss, te_nb = 0.0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in test_dl:
                if cfg.model == "context":
                    z_win, a_true = batch[0].to(device), batch[1].to(device)
                    a_pred = model(z_win)
                else:
                    zt, a_true, zt1 = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    a_pred = model(zt, zt1)
                te_loss += F.mse_loss(a_pred, a_true).item()
                te_nb += 1
                all_pred.append(a_pred.cpu())
                all_true.append(a_true.cpu())

        tr = ep_loss / max(n_b, 1)
        te = te_loss / max(te_nb, 1)
        train_losses.append(tr)
        test_losses.append(te)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[IDM] epoch {epoch:03d}/{cfg.epochs} train {tr:.6f} test {te:.6f} | {time.time()-t0:.1f}s")

    metrics = compute_metrics(torch.cat(all_pred), torch.cat(all_true))
    print(f"\n[IDM] Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    torch.save(model.state_dict(), run_dir / "model.pth")
    torch.save({"train": train_losses, "test": test_losses, "metrics": metrics}, run_dir / "results.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, lw=2, label="train", color="#2d6cdf")
    ax.plot(test_losses, lw=2, label="test", color="#e67e22")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"IDM ({cfg.model})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", dpi=140)
    plt.close()

    print(f"[IDM] saved → {run_dir}")
    return run_dir, metrics
