
import numpy as np
import torch
import gymnasium as gym
from PIL import Image

from Autoencoder.network import AutoEncoder


def _frame_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(rgb).convert("L").resize((84, 84), Image.BILINEAR)
    return torch.tensor(np.array(img, dtype=np.float32) / 255.0)


def collect_rollouts(
    policy, ae: AutoEncoder, device: torch.device,
    n_episodes: int = 50, max_steps: int = 1000,
) -> dict:
    ae.eval()
    policy.eval()
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")

    all_frames, all_actions, boundaries = [], [], [0]
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_frames, ep_actions = [], []

        for step in range(max_steps):
            rgb = env.render()
            frame = _frame_to_tensor(rgb)
            ep_frames.append(frame)

            with torch.no_grad():
                z = ae.encoder(frame.unsqueeze(0).unsqueeze(0).to(device))
                action = policy(z).cpu().squeeze(0).numpy()

            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_actions.append(torch.tensor(action, dtype=torch.float32))
            ep_reward += reward

            if terminated or truncated:
                break

        all_frames.extend(ep_frames)
        all_actions.extend(ep_actions)
        boundaries.append(boundaries[-1] + len(ep_frames))
        rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            print(f"  [rollout] {ep+1}/{n_episodes} eps, "
                  f"avg reward {np.mean(rewards[-10:]):.1f}")

    env.close()

    return {
        "frames": torch.stack(all_frames),
        "actions": torch.stack(all_actions),
        "episode_boundaries": torch.tensor(boundaries),
        "rewards": rewards,
    }
