"""Config for IDM."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

STUDENT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class IDMConfig:
    model: str = "pair"
    z_dim: int = 64
    action_dim: int = 2
    hidden_dim: int = 256
    context_len: int = 8
    n_heads: int = 4
    num_layers: int = 2
    test_ratio: float = 0.2
    seed: int = 42
    batch_size: int = 256
    epochs: int = 30
    lr: float = 3e-4
    grad_clip: float = 1.0
    device: Optional[str] = None

    data_dir: Path = STUDENT_DIR / "data" / "labeled_rollouts"
    runs_dir: Path = STUDENT_DIR / "runs" / "idm"
    ae_runs_dir: Path = STUDENT_DIR / "runs" / "autoencoder"
