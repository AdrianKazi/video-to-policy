"""Config for BCO."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

STUDENT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class BCOConfig:
    state_dim: int = 8
    action_dim: int = 2
    hidden_dim: int = 256

    # IDM training
    idm_epochs: int = 50
    idm_lr: float = 3e-4
    idm_batch_size: int = 256

    # BC policy training
    bc_epochs: int = 100
    bc_lr: float = 3e-4
    bc_batch_size: int = 256

    # data collection
    n_random_episodes: int = 5000
    n_rollout_episodes: int = 200
    n_eval_episodes: int = 50
    max_steps: int = 1000

    # BCO loop
    n_iterations: int = 10
    patience: int = 2

    test_ratio: float = 0.2
    seed: int = 42
    grad_clip: float = 1.0
    device: Optional[str] = None

    data_dir: Path = STUDENT_DIR / "data" / "labeled_rollouts"
    runs_dir: Path = STUDENT_DIR / "runs" / "bco"
