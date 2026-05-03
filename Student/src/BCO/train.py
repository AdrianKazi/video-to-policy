
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset

from BCO.config import BCOConfig
from BCO.network import StateBCPolicy, StateIDM
from IDM.metrics import compute_metrics


def _collect_random_rollouts(n_episodes: int, max_steps: int = 1000):
    env = gym.make("LunarLanderContinuous-v3")
    states, actions, bounds = [], [], [0]
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_s, ep_a = [], []
        for _ in range(max_steps):
            action = env.action_space.sample()
            ep_s.append(obs.copy())
            ep_a.append(action.copy())
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        states.extend(ep_s)
        actions.extend(ep_a)
        bounds.append(bounds[-1] + len(ep_s))
    env.close()
    return (torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32), bounds)


def _collect_policy_rollouts(policy: StateBCPolicy, n_episodes: int,
                             max_steps: int = 1000):
    policy.eval()
    env = gym.make("LunarLanderContinuous-v3")
    states, actions, bounds, rewards = [], [], [0], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_s, ep_a, ep_r = [], [], 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                action = policy(torch.FloatTensor(obs).unsqueeze(0)).numpy()[0]
            ep_s.append(obs.copy())
            ep_a.append(action.copy())
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if term or trunc:
                break
        states.extend(ep_s)
        actions.extend(ep_a)
        bounds.append(bounds[-1] + len(ep_s))
        rewards.append(ep_r)
    env.close()
    return (torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            bounds, rewards)


def _build_pairs(states, actions, bounds):
    st, st1, at = [], [], []
    for ep in range(len(bounds) - 1):
        s, e = bounds[ep], bounds[ep + 1]
        if e - s < 2:
            continue
        st.append(states[s:e-1])
        st1.append(states[s+1:e])
        at.append(actions[s:e-1])
    return torch.cat(st), torch.cat(st1), torch.cat(at)


def _train_state_idm(st, st1, at, cfg: BCOConfig):
    idm = StateIDM(cfg.state_dim, cfg.action_dim, cfg.hidden_dim)
    dl = DataLoader(TensorDataset(st, st1, at),
                    batch_size=cfg.idm_batch_size, shuffle=True)
    opt = torch.optim.Adam(idm.parameters(), lr=cfg.idm_lr)

    for epoch in range(cfg.idm_epochs):
        idm.train()
        for s, s1, a in dl:
            loss = F.mse_loss(idm(s, s1), a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(idm.parameters(), cfg.grad_clip)
            opt.step()
    return idm


def _pseudo_label_expert(idm: StateIDM, expert_states, expert_actions, expert_bounds):
    idm.eval()
    pl_s, pl_a, pl_true = [], [], []
    with torch.no_grad():
        for ep in range(len(expert_bounds) - 1):
            s, e = expert_bounds[ep], expert_bounds[ep + 1]
            if e - s < 2:
                continue
            st, st1 = expert_states[s:e-1], expert_states[s+1:e]
            pl_s.append(st)
            pl_a.append(idm(st, st1))
            pl_true.append(expert_actions[s:e-1])
    return torch.cat(pl_s), torch.cat(pl_a), torch.cat(pl_true)


def _train_bc(states, actions, cfg: BCOConfig):
    bc = StateBCPolicy(cfg.state_dim, cfg.action_dim, cfg.hidden_dim)
    n = len(states)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))
    n_test = int(n * cfg.test_ratio)

    train_dl = DataLoader(
        TensorDataset(states[idx[n_test:]], actions[idx[n_test:]]),
        batch_size=cfg.bc_batch_size, shuffle=True)

    opt = torch.optim.Adam(bc.parameters(), lr=cfg.bc_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.bc_epochs)

    for epoch in range(1, cfg.bc_epochs + 1):
        bc.train()
        ep_loss, n_b = 0.0, 0
        for sb, ab in train_dl:
            loss = F.mse_loss(bc(sb), ab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += loss.item()
            n_b += 1
        sched.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [BC] epoch {epoch:03d}/{cfg.bc_epochs} "
                  f"train {ep_loss / max(n_b, 1):.6f}")
    return bc


def _eval_bc(bc: StateBCPolicy, n_episodes: int, max_steps: int = 1000):
    bc.eval()
    env = gym.make("LunarLanderContinuous-v3")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                action = bc(torch.FloatTensor(obs).unsqueeze(0)).numpy()[0]
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if term or trunc:
                break
        rewards.append(ep_r)
    env.close()
    return rewards


def run_bco(cfg: BCOConfig, expert_data_path: Path | None = None) -> Path:
    t0 = time.time()

    # load expert (state, action) trajectories
    if expert_data_path is None:
        expert_data_path = cfg.data_dir / "expert_with_states.pt"
    expert = torch.load(expert_data_path, map_location="cpu")
    expert_states = expert["states"]
    expert_actions = expert["actions"]
    expert_bounds = expert["episode_boundaries"].tolist()
    print(f"[BCO] {len(expert_states)} expert frames, {len(expert_bounds)-1} episodes")

    # collect random exploration data
    print(f"[BCO] Collecting {cfg.n_random_episodes} random rollouts...")
    rand_s, rand_a, rand_b = _collect_random_rollouts(
        cfg.n_random_episodes, cfg.max_steps)
    print(f"  {len(rand_s)} state-action pairs")

    stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_dir = cfg.runs_dir / f"bco_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    idm_s, idm_a, idm_b = rand_s, rand_a, rand_b
    results = []
    best_bc, best_reward = None, -float("inf")

    for it in range(cfg.n_iterations):
        print(f"\n{'='*60}")
        print(f"[BCO] Iteration {it+1}/{cfg.n_iterations}")
        print(f"{'='*60}")

        # train IDM
        st, st1, at = _build_pairs(idm_s, idm_a, idm_b)
        print(f"\n  Training IDM on {len(st)} transition pairs...")
        idm = _train_state_idm(st, st1, at, cfg)

        # pseudo-label expert
        pl_s, pl_a, pl_true = _pseudo_label_expert(
            idm, expert_states, expert_actions, expert_bounds)
        pl_metrics = compute_metrics(pl_a, pl_true)
        print(f"  Pseudo-label: R²={pl_metrics['r2']:.3f}, "
              f"cos={pl_metrics['cosine_sim']:.3f}, MSE={pl_metrics['mse']:.4f}")

        # train BC
        print(f"\n  Training BC on {len(pl_s)} pseudo-labeled pairs...")
        bc = _train_bc(pl_s, pl_a, cfg)

        # evaluate
        rews = _eval_bc(bc, cfg.n_eval_episodes, cfg.max_steps)
        avg_r, std_r = np.mean(rews), np.std(rews)
        print(f"\n  Reward: {avg_r:.1f} +/- {std_r:.1f}")

        if avg_r > best_reward:
            best_reward = avg_r
            best_bc = bc

        result = {
            "iteration": it + 1,
            "pl_metrics": pl_metrics,
            "reward_mean": avg_r,
            "reward_std": std_r,
        }
        results.append(result)

        # save iteration checkpoint
        it_dir = run_dir / f"iter_{it+1}"
        it_dir.mkdir(exist_ok=True)
        torch.save(bc.state_dict(), it_dir / "policy.pth")
        torch.save(idm.state_dict(), it_dir / "idm.pth")

        # collect on-policy data for IDM retraining
        if it < cfg.n_iterations - 1:
            print(f"\n  Collecting {cfg.n_rollout_episodes} on-policy rollouts...")
            op_s, op_a, op_b, op_r = _collect_policy_rollouts(
                bc, cfg.n_rollout_episodes, cfg.max_steps)
            print(f"  {len(op_s)} pairs, avg reward {np.mean(op_r):.1f}")

            # combine: all random + on-policy
            offset = len(rand_s)
            combined_b = list(rand_b) + [b + offset for b in op_b[1:]]
            idm_s = torch.cat([rand_s, op_s])
            idm_a = torch.cat([rand_a, op_a])
            idm_b = combined_b

    # save best policy
    torch.save(best_bc.state_dict(), run_dir / "best_policy.pth")
    torch.save(results, run_dir / "results.pt")
    _plot_results(results, run_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"[BCO] Done in {elapsed:.0f}s → {run_dir}")
    print(f"{'='*60}")
    print(f"{'Iter':>4} | {'PL R²':>7} | {'Reward':>12}")
    print("-" * 35)
    for r in results:
        print(f"{r['iteration']:>4} | {r['pl_metrics']['r2']:>7.3f} | "
              f"{r['reward_mean']:>7.1f} +/- {r['reward_std']:.1f}")
    print(f"\nBest: {best_reward:.1f} | Teacher: ~261")

    return run_dir


def _plot_results(results: list[dict], run_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    iters = [r["iteration"] for r in results]
    rewards = [r["reward_mean"] for r in results]
    stds = [r["reward_std"] for r in results]
    axes[0].errorbar(iters, rewards, yerr=stds, marker="o", capsize=4,
                     lw=2, color="#2d6cdf")
    axes[0].axhline(y=261, color="green", ls="--", lw=1, label="Teacher")
    axes[0].axhline(y=-205, color="red", ls="--", lw=1, label="Random")
    axes[0].set_xlabel("BCO iteration")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("BC Policy Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    r2s = [r["pl_metrics"]["r2"] for r in results]
    axes[1].plot(iters, r2s, marker="s", lw=2, color="#e67e22")
    axes[1].set_xlabel("BCO iteration")
    axes[1].set_ylabel("R2")
    axes[1].set_title("Pseudo-label Quality (vs true actions)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir / "bco_progress.png", dpi=140)
    plt.close()
