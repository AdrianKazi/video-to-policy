# Implementation here is very much based on,
# https://github.com/warrenzha/ppo-gae-pytorch/blob/main/agent/ppo_discrete.py
# https://github.com/warrenzha/ppo-gae-pytorch/blob/main/agent/ppo_continous.py

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from models.autoencoder import AutoEncoder
from models.discriminator import Discriminator
from models.policy import PolicyNetwork, ValueNetwork
from config.gaifo_config import *
from config.paths import RUNS_DIR, DATASETS_DIR
from data_processing.build_expert_transitions import find_latest_autoencoder

class FrameEncoder:
    def __init__(self, autoencoder, device="cpu"):
        self.ae = autoencoder
        self.ae.eval()
        self.device = device

    @torch.no_grad()
    def encode_frame(self, rgb_frame):
        """
        Encode an RGB frame (H, W, 3) from the environment into z_dim latent.
        Matches the preprocessing in extract_frames.py: grayscale, resize to 84x84.
        """
        import cv2
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84))
        x = torch.from_numpy(gray).float() / 255.0
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 84, 84)
        z = self.ae.encoder(x)   
        return z.squeeze(0) 



class RolloutBuffer:
    """Stores transitions from policy rollouts for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.z_current = []
        self.z_next = []

    def add(self, state, action, log_prob, reward, done, value, z_t, z_t1):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.z_current.append(z_t)
        self.z_next.append(z_t1)

    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs).squeeze(-1),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
            torch.stack(self.values).squeeze(-1),
            torch.stack(self.z_current),
            torch.stack(self.z_next),
        )

    def clear(self):
        self.__init__()



def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def train_gaifo(autoencoder_path=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    if autoencoder_path is None:
        autoencoder_path = find_latest_autoencoder(None)

    ae = AutoEncoder(z_dim=Z_DIM)
    ae.load_state_dict(torch.load(autoencoder_path, weights_only=True))
    ae.to(device)
    frame_encoder = FrameEncoder(ae, device)
    print(f"[AE] Loaded from {autoencoder_path}")

    expert_path = os.path.join(DATASETS_DIR, "expert_transitions.pt")
    expert_transitions = torch.load(expert_path, weights_only=True).to(device)
    expert_z_t = expert_transitions[:, 0]   # (N, z_dim)
    expert_z_t1 = expert_transitions[:, 1]  # (N, z_dim)
    print(f"[EXPERT] {expert_transitions.shape[0]} transitions loaded")

    discriminator = Discriminator(Z_DIM, DISC_HIDDEN_DIM).to(device)
    policy = PolicyNetwork(STATE_DIM, ACTION_DIM, MAX_ACTION, POLICY_HIDDEN_DIM).to(device)
    value_net = ValueNetwork(STATE_DIM, POLICY_HIDDEN_DIM).to(device)

    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=DISC_LR)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=POLICY_LR)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=VALUE_LR)

    bce_loss = nn.BCELoss()

    env = gym.make(ENV_NAME, render_mode="rgb_array")

    disc_losses_history = []
    disc_acc_history = []
    policy_losses_history = []
    eval_rewards_history = []
    landing_success_history = []
    reward_variance_history = []
    best_eval_reward = -float("inf")

    from datetime import datetime
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"gaifo_policy_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Copy artifacts for reproducibility
    import shutil
    shutil.copy(autoencoder_path, os.path.join(run_dir, "autoencoder.pth"))
    shutil.copy(expert_path, os.path.join(run_dir, "expert_transitions.pt"))
    config_src = os.path.join(os.path.dirname(__file__), "../config/gaifo_config.py")
    shutil.copy(config_src, os.path.join(run_dir, "gaifo_config.py"))
    print(f"[SAVED] Copied autoencoder, expert transitions, and config to {run_dir}")

    print(f"\n[TRAIN GAIfO] Starting {TOTAL_ITERATIONS} iterations\n")

    for iteration in range(1, TOTAL_ITERATIONS + 1):

    
        buffer = RolloutBuffer()
        state, _ = env.reset()
        ep_steps = 0

        for step in range(ROLLOUT_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            rgb_frame = env.render()
            z_t = frame_encoder.encode_frame(rgb_frame)

            with torch.no_grad():
                action, log_prob = policy.get_action(state_tensor)
                value = value_net(state_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            next_state, _, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            ep_steps += 1

            rgb_frame_next = env.render()
            z_t1 = frame_encoder.encode_frame(rgb_frame_next)

            buffer.add(
                state_tensor.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
                0.0,
                float(done),
                value.squeeze(0),
                z_t,
                z_t1,
            )

            if done or ep_steps >= MAX_EP_STEPS:
                state, _ = env.reset()
                ep_steps = 0
            else:
                state = next_state

       
        (states, actions, old_log_probs, _, dones, old_values,
         learner_z_t, learner_z_t1) = buffer.get()

        learner_z_t = learner_z_t.to(device)
        learner_z_t1 = learner_z_t1.to(device)

        disc_loss_avg = 0

        for d_epoch in range(DISC_EPOCHS):
            n_expert = expert_z_t.shape[0]
            n_learner = learner_z_t.shape[0]
            batch_size = min(DISC_BATCH_SIZE, n_expert, n_learner)

            expert_idx = torch.randint(0, n_expert, (batch_size,))
            learner_idx = torch.randint(0, n_learner, (batch_size,))

            expert_pred = discriminator(
                expert_z_t[expert_idx], expert_z_t1[expert_idx]
            )
            learner_pred = discriminator(
                learner_z_t[learner_idx], learner_z_t1[learner_idx]
            )

            expert_labels = torch.zeros_like(expert_pred)
            learner_labels = torch.ones_like(learner_pred)

            loss_expert = bce_loss(expert_pred, expert_labels)
            loss_learner = bce_loss(learner_pred, learner_labels)
            disc_loss = loss_expert + loss_learner

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            disc_loss_avg += disc_loss.item()

        disc_loss_avg /= DISC_EPOCHS
        disc_losses_history.append(disc_loss_avg)

        # Compute discriminator accuracy
        with torch.no_grad():
            n_acc = min(512, expert_z_t.shape[0], learner_z_t.shape[0])
            acc_expert_idx = torch.randint(0, expert_z_t.shape[0], (n_acc,))
            acc_learner_idx = torch.randint(0, learner_z_t.shape[0], (n_acc,))
            expert_acc = (discriminator(expert_z_t[acc_expert_idx], expert_z_t1[acc_expert_idx]) < 0.5).float().mean().item()
            learner_acc = (discriminator(learner_z_t[acc_learner_idx], learner_z_t1[acc_learner_idx]) > 0.5).float().mean().item()
            disc_acc = (expert_acc + learner_acc) / 2
        disc_acc_history.append(disc_acc)

        with torch.no_grad():
            gaifo_rewards = discriminator.reward(
                learner_z_t, learner_z_t1
            ).squeeze(-1).cpu()

     
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        dones = dones.to(device)

        with torch.no_grad():
            new_values = value_net(states).squeeze(-1).cpu()

        advantages, returns = compute_gae(gaifo_rewards, new_values, dones.cpu())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.to(device)
        advantages = advantages.to(device)

        policy_loss_avg = 0

        for ppo_epoch in range(PPO_EPOCHS):
            indices = torch.randperm(len(states))

            for start in range(0, len(states), PPO_BATCH_SIZE):
                end = start + PPO_BATCH_SIZE
                idx = indices[start:end]

                new_log_probs, entropy = policy.evaluate_actions(
                    states[idx], actions[idx]
                )

                ratio = torch.exp(new_log_probs.squeeze(-1) - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                total_policy_loss = policy_loss + ENTROPY_COEFF * entropy_loss

                policy_optimizer.zero_grad()
                total_policy_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                policy_optimizer.step()

                new_values = value_net(states[idx]).squeeze(-1)
                value_loss = VALUE_LOSS_COEFF * (returns[idx] - new_values).pow(2).mean()

                value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), MAX_GRAD_NORM)
                value_optimizer.step()

                policy_loss_avg += total_policy_loss.item()

        n_batches = PPO_EPOCHS * (len(states) // PPO_BATCH_SIZE + 1)
        policy_loss_avg /= max(n_batches, 1)
        policy_losses_history.append(policy_loss_avg)

        if iteration % LOG_INTERVAL == 0:
            avg_reward = gaifo_rewards.mean().item()
            print(
                f"[Iter {iteration:>4}/{TOTAL_ITERATIONS}] "
                f"Disc loss: {disc_loss_avg:.4f} | "
                f"Disc acc: {disc_acc:.1%} | "
                f"Policy loss: {policy_loss_avg:.4f} | "
                f"Avg GAIfO reward: {avg_reward:.4f}"
            )

        if iteration % EVAL_INTERVAL == 0:
            eval_reward, eval_var, success_rate = evaluate_policy(policy, device)
            eval_rewards_history.append((iteration, eval_reward))
            reward_variance_history.append((iteration, eval_var))
            landing_success_history.append((iteration, success_rate))
            print(
                f"  [EVAL] Iter {iteration} → Avg reward: {eval_reward:.2f} | "
                f"Variance: {eval_var:.2f} | Landing: {success_rate:.0%}"
            )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(policy.state_dict(), os.path.join(run_dir, "model.pth"))
                torch.save(value_net.state_dict(), os.path.join(run_dir, "value_net.pth"))
                torch.save(discriminator.state_dict(), os.path.join(run_dir, "discriminator.pth"))
                print(f"  [BEST] New best: {best_eval_reward:.2f} — saved!")

    env.close()

    save_training_plots(
        disc_losses_history,
        disc_acc_history,
        policy_losses_history,
        eval_rewards_history,
        landing_success_history,
        reward_variance_history,
        run_dir
    )

    print(f"\n[DONE] Best eval reward: {best_eval_reward:.2f}")
    print(f"[SAVED] {run_dir}")

    return policy, run_dir

def evaluate_policy(policy, device, n_episodes=EVAL_EPISODES):
    """Run the policy greedily and return avg reward, variance, and landing success rate."""
    eval_env = gym.make(ENV_NAME)
    episode_rewards = []
    landings = 0

    policy.eval()
    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        done = False
        ep_reward = 0
        steps = 0
        landed = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = policy.get_action(state_tensor, deterministic=True)
            state, reward, terminated, truncated, _ = eval_env.step(
                action.squeeze(0).cpu().numpy()
            )
            if terminated:
                landed = True
            done = terminated or truncated
            ep_reward += reward
            steps += 1
            if steps >= MAX_EP_STEPS:
                break

        episode_rewards.append(ep_reward)
        if landed and ep_reward > 0:
            landings += 1

    eval_env.close()
    policy.train()

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    variance = np.var(episode_rewards)
    success_rate = landings / n_episodes
    return avg_reward, variance, success_rate

def save_training_plots(disc_losses, disc_accs, policy_losses, eval_rewards,
                        landing_success, reward_variance, run_dir):
    """Save training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(disc_losses)
    axes[0, 0].set_title("Discriminator Loss")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("BCE Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(disc_accs)
    axes[0, 1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% (equilibrium)")
    axes[0, 1].set_title("Discriminator Accuracy")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(policy_losses)
    axes[0, 2].set_title("Policy Loss (PPO)")
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].grid(True)

    if eval_rewards:
        iters, rewards = zip(*eval_rewards)
        axes[1, 0].plot(iters, rewards, marker="o")
        axes[1, 0].set_title("Evaluation Reward")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Avg Reward")
        axes[1, 0].grid(True)

    if landing_success:
        iters, rates = zip(*landing_success)
        axes[1, 1].plot(iters, [r * 100 for r in rates], marker="s", color="green")
        axes[1, 1].axhline(y=80, color="r", linestyle="--", alpha=0.5, label="80% target")
        axes[1, 1].set_title("Landing Success Rate")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Success %")
        axes[1, 1].set_ylim(0, 105)
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    if reward_variance:
        iters, variances = zip(*reward_variance)
        axes[1, 2].plot(iters, variances, marker="^", color="orange")
        axes[1, 2].set_title("Reward Variance")
        axes[1, 2].set_xlabel("Iteration")
        axes[1, 2].set_ylabel("Var(episode rewards)")
        axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "gaifo_training.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved training curves → {run_dir}/gaifo_training.png")


if __name__ == "__main__":
    train_gaifo()
