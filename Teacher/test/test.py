# test/test.py

import os
import csv
import torch
import cv2
import numpy as np

from gymnasium.wrappers import RecordVideo

from env.env import make_env, get_env_dims
from agents.td3 import TD3_Agent
from config.config import *
from tracking.mlflow_utils import setup_mlflow, start_run
import mlflow


def save_episode_npz(trajectory_dir, episode_idx, states, actions, next_states, rewards, terminateds, truncateds, frames, next_frames, ep_reward,):
    """
    Save one expert rollout as a compressed .npz file.

    Each row i corresponds to one env transition:

        states[i] -- actions[i] --> next_states[i]

    This is the supervised data that can be used for inverse dynamics:
        f(s_t, s_{t+1}) -> a_t

    On the Student side, we'll have:
        f(z_t, z_{t+1}) -> a_t
    """

    os.makedirs(trajectory_dir, exist_ok=True)

    dones = np.logical_or(terminateds, truncateds)

    file_name = f"expert_episode_{episode_idx:03d}.npz"
    file_path = os.path.join(trajectory_dir, file_name)

    np.savez_compressed(
        file_path,
        episode_idx=np.array([episode_idx], dtype=np.int32),
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminateds=np.asarray(terminateds, dtype=np.bool_),
        truncateds=np.asarray(truncateds, dtype=np.bool_),
        dones=np.asarray(dones, dtype=np.bool_),
        frames=np.asarray(frames, dtype=np.uint8),
        next_frames=np.asarray(next_frames, dtype=np.uint8),
        ep_reward=np.array([ep_reward], dtype=np.float32),
        length=np.array([len(actions)], dtype=np.int32),
    )

    return file_name, file_path


def append_manifest_row(manifest_path, row):
    """
    Keep a small CSV index so the Student pipeline can discover episodes
    without scanning and guessing.
    """

    file_exists = os.path.exists(manifest_path)

    fieldnames = ["episode_idx", "trajectory_file", "num_steps", "episode_reward"]

    with open(manifest_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def test(num_episodes=20, video_path="videos", trajectory_dir="trajectories", model_path="models_saved/actor.pth", save_trajectories=True):
    """
    Optionally saves expert transition trajectories with the true expert actions (useful for action inference).

    """

    os.makedirs(video_path, exist_ok=True)

    if save_trajectories:
        os.makedirs(trajectory_dir, exist_ok=True)

    manifest_path = os.path.join(trajectory_dir, "manifest.csv")

    env = make_env(render_mode="rgb_array")

    env = RecordVideo(env, video_path, episode_trigger=lambda e: True, name_prefix="expert")

    state_dim, action_dim, max_action = get_env_dims(env)
    agent = TD3_Agent(state_dim, action_dim, max_action)

    agent.actor.load_state_dict(torch.load(model_path))
    agent.actor.eval()

    setup_mlflow()

    with start_run("test"):

        for episode in range(num_episodes):

            state, _ = env.reset()
            ep_reward = 0.0
            done = False

            states = []
            actions = []
            next_states = []
            rewards = []
            terminateds = []
            truncateds = []
            frames = []
            next_frames = []

            while not done:
                frame = env.render()
                frame = cv2.resize(frame, (84, 84)).astype(np.uint8)

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = agent.actor(state_tensor).detach().cpu().numpy()[0]

                next_state, reward, terminated, truncated, _ = env.step(action)
                next_frame = env.render()
                next_frame = cv2.resize(next_frame, (84, 84)).astype(np.uint8)

                done = terminated or truncated

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                terminateds.append(terminated)
                truncateds.append(truncated)
                frames.append(frame)
                next_frames.append(next_frame)

                state = next_state
                ep_reward += reward

            num_steps = len(actions)

            if save_trajectories:
                trajectory_file, trajectory_path = save_episode_npz(
                    trajectory_dir=trajectory_dir,
                    episode_idx=episode,
                    states=states,
                    actions=actions,
                    next_states=next_states,
                    rewards=rewards,
                    terminateds=terminateds,
                    truncateds=truncateds,
                    frames=frames,
                    next_frames=next_frames,
                    ep_reward=ep_reward,
                )

                append_manifest_row(
                    manifest_path,
                    {
                        "episode_idx": episode,
                        "trajectory_file": trajectory_file,
                        "num_steps": num_steps,
                        "episode_reward": ep_reward,
                    },
                )

            print(f"Test episode {episode:03d} | " f"steps: {num_steps:04d} | " f"reward: {ep_reward:.2f}")

            mlflow.log_metric("test_reward", ep_reward, step=episode)
            mlflow.log_metric("test_length", num_steps, step=episode)

    env.close()