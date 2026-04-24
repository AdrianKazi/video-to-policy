# Ref: https://github.com/warrenzha/ppo-gae-pytorch/blob/main/test.py

import os
import sys
import shutil
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from models.policy import PolicyNetwork
from config.gaifo_config import ENV_NAME, STATE_DIM, ACTION_DIM, MAX_ACTION, POLICY_HIDDEN_DIM
from config.paths import RUNS_DIR


def find_latest_gaifo_run():
    gaifo_runs = [d for d in os.listdir(RUNS_DIR) if d.startswith("gaifo_policy_")]
    if not gaifo_runs:
        raise FileNotFoundError(f"No GAIfO runs found in {RUNS_DIR}")
    gaifo_runs.sort()

    return os.path.join(RUNS_DIR, gaifo_runs[-1])


def evaluate(run_dir=None, n_episodes=10, record_video=True):
    if run_dir is None:
        run_dir = find_latest_gaifo_run()

    model_path = os.path.join(run_dir, "model.pth")
    print(f"[EVAL] Loading policy from {model_path}")

    config_src = os.path.join(os.path.dirname(__file__), "../../config/gaifo_config.py")
    shutil.copy(config_src, os.path.join(run_dir, "gaifo_config.py"))

    policy = PolicyNetwork(STATE_DIM, ACTION_DIM, MAX_ACTION, POLICY_HIDDEN_DIM)
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()

    # Create env
    if record_video:
        video_dir = os.path.join(run_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.make(ENV_NAME, render_mode="rgb_array")
        env = RecordVideo(env, video_dir, episode_trigger=lambda e: True)
    else:
        env = gym.make(ENV_NAME)

    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.get_action(state_tensor, deterministic=True)

            action_np = action.squeeze(0).cpu().numpy()
            state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward

        rewards.append(ep_reward)
        print(f"  Episode {episode:>2} | Reward: {ep_reward:.2f}")

    env.close()

    avg_reward = sum(rewards) / len(rewards)
    print(f"\n[RESULT] Avg reward over {n_episodes} episodes: {avg_reward:.2f}")
    print(f"[RESULT] Min: {min(rewards):.2f} | Max: {max(rewards):.2f}")  

    if record_video:
        print(f"[VIDEOS] Saved to {video_dir}")

    return rewards


if __name__ == "__main__":
    evaluate()
