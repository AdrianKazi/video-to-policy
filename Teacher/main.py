import argparse
from train.train import train
from test.test import test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of evaluation episodes to run in test mode.")
    parser.add_argument("--video_path", type=str, default="videos", help="Directory where expert rollout videos are saved.")
    parser.add_argument("--trajectory_dir", type=str, default="trajectories", help="Directory where expert transition trajectories are saved.")
    parser.add_argument("--model_path", type=str, default="models_saved/actor.pth", help="Path to trained Teacher actor checkpoint.")
    parser.add_argument("--no_save_trajectories", action="store_true", help="Disable saving .npz transition trajectories during test.")

    args = parser.parse_args()

    if args.mode == "train":
        print("START TRAIN")   # DEBUG
        train()

    elif args.mode == "test":
        print("START TEST")
        test(num_episodes=args.num_episodes, video_path=args.video_path, trajectory_dir=args.trajectory_dir, model_path=args.model_path, save_trajectories=not args.no_save_trajectories)


if __name__ == "__main__":
    main()