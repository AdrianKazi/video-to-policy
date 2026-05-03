
import argparse
from pathlib import Path

from IDM.config import IDMConfig
from IDM.train import run_train


def _apply_overrides(cfg: IDMConfig, args: argparse.Namespace) -> None:
    if getattr(args, "model", None) is not None:
        cfg.model = args.model
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "lr", None) is not None:
        cfg.lr = args.lr
    if getattr(args, "context_len", None) is not None:
        cfg.context_len = args.context_len
    if getattr(args, "device", None) is not None:
        cfg.device = args.device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IDM pipeline")
    parser.add_argument("--model", type=str, default=None, choices=["pair", "context"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--ae-checkpoint", type=Path, default=None)

    args = parser.parse_args(argv)
    cfg = IDMConfig()
    _apply_overrides(cfg, args)

    if args.cmd == "train":
        run_train(cfg, ae_checkpoint=getattr(args, "ae_checkpoint", None))
