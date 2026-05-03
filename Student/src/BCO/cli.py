
import argparse
from pathlib import Path

from BCO.config import BCOConfig
from BCO.train import run_bco


def _apply_overrides(cfg: BCOConfig, args: argparse.Namespace) -> None:
    for attr in ("bc_epochs", "bc_lr", "idm_epochs", "n_iterations",
                 "n_rollout_episodes", "n_random_episodes",
                 "n_eval_episodes", "device"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BCO pipeline")
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--bc-lr", type=float, default=None)
    parser.add_argument("--idm-epochs", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--n-rollout-episodes", type=int, default=None)
    parser.add_argument("--n-random-episodes", type=int, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--expert-data", type=Path, default=None)

    args = parser.parse_args(argv)
    cfg = BCOConfig()
    _apply_overrides(cfg, args)

    if args.cmd == "run":
        run_bco(cfg, expert_data_path=getattr(args, "expert_data", None))
