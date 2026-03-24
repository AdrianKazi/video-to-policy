import os
from datetime import datetime
import torch
from config.paths import RUNS_DIR


def save_model(model, losses, model_name="model"):
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')

    run_dir = os.path.join(RUNS_DIR, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))
    torch.save(losses, os.path.join(run_dir, "losses.pt"))

    print(f"[SAVED] {run_dir}")

    return run_dir