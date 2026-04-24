import os
import sys
import argparse

from config.paths import DATASETS_DIR, RUNS_DIR


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument("--skip-ae", action="store_true",
                        help="Skip autoencoder training (use latest avaialable)")
    parser.add_argument("--skip-transitions", action="store_true",
                        help="Skip expert transition extraction (use latest avaialable)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate the latest trained policy")
    args = parser.parse_args()

    if args.eval_only:
        print("\n" + "=" * 50)
        print("  EVALUATE GAIFO POLICY")
        print("=" * 50 + "\n")
        from evaluation.gaifo.evaluate_gaifo import evaluate
        evaluate()
        return

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

    # =========================
    # 3. Main policy training loop
    # References: 
    # https://github.com/warrenzha/ppo-gae-pytorch/blob/main/train.py#L51
    # =========================
    print("\n" + "=" * 50)
    print("  STEP 3: Train GAIfO Policy")
    print("=" * 50 + "\n")

    from training.train_gaifo import train_gaifo
    policy, run_dir = train_gaifo()

    # =========================
    # 4. Evaluate
    # Reference: https://github.com/warrenzha/ppo-gae-pytorch/blob/97f7bb338227321b218bea5aa6a16bcc23c8618e/train.py#L23
    # =========================
    print("\n" + "=" * 50)
    print("  STEP 4: Evaluate Trained Policy")
    print("=" * 50 + "\n")

    from evaluation.gaifo.evaluate_gaifo import evaluate
    evaluate(run_dir=run_dir)

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
