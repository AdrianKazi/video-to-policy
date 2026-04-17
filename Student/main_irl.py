import os
import sys
import argparse

from config.paths import DATASETS_DIR, RUNS_DIR


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument("--skip-ae", action="store_true",
                        help="Skip autoencoder training (use existing)")
    parser.add_argument("--skip-transitions", action="store_true",
                        help="Skip expert transition extraction (use existing)")
    args = parser.parse_args()

    if not args.skip_ae:
        print("\n" + "=" * 50)
        print("  STEP 1: Train AutoEncoder")
        print("=" * 50 + "\n")

        from data_processing.extract_frames import extract_frames
        from data_processing.build_dataset import build_dataset
        from data_processing.load_dataset import DownloadVideoDataset
        from training.train_autoencoder import train_autoencoder
        from models.autoencoder import AutoEncoder

        extract_frames()
        build_dataset()
        train_path = os.path.join(DATASETS_DIR, "train.pt")
        train_dataset = DownloadVideoDataset(train_path)
        train_autoencoder(train_dataset, AutoEncoder)
    else:
        print("\n[SKIP] AutoEncoder training (using existing)\n")

    # =========================
    # 2. Build expert transitions
    # =========================
    transitions_path = os.path.join(DATASETS_DIR, "expert_transitions.pt")

    if not args.skip_transitions or not os.path.exists(transitions_path):
        print("\n" + "=" * 50)
        print("  STEP 2: Build Expert Transitions")
        print("=" * 50 + "\n")

        from data_processing.build_expert_transitions import build_expert_transitions
        build_expert_transitions()
    else:
        print(f"\n[SKIP] Expert transitions (using {transitions_path})\n")


if __name__ == "__main__":
    main()
