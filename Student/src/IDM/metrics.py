
import torch
import torch.nn.functional as F


def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(pred, true).item()
    mae = (pred - true).abs().mean(dim=0)
    cos = F.cosine_similarity(pred, true, dim=-1).mean().item()
    ss_res = ((true - pred) ** 2).sum().item()
    ss_tot = ((true - true.mean(dim=0)) ** 2).sum().item()
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return {
        "mse": mse,
        "mae_main": mae[0].item(),
        "mae_side": mae[1].item(),
        "cosine_sim": cos,
        "r2": r2,
    }
