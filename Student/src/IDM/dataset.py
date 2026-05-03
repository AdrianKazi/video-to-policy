
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, z: torch.Tensor, actions: torch.Tensor):
        self.z_t = z[:-1]
        self.z_t1 = z[1:]
        self.actions = actions[:-1]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.z_t[idx], self.actions[idx], self.z_t1[idx]


class ContextDataset(Dataset):
    def __init__(self, z: torch.Tensor, actions: torch.Tensor, context_len: int):
        self.z = z
        self.actions = actions
        self.k = context_len

    def __len__(self):
        return len(self.actions) - self.k

    def __getitem__(self, idx):
        t = idx + self.k
        return self.z[t - self.k : t + 1], self.actions[t - 1]


def encode_frames(frames: torch.Tensor, ae, device, batch_size: int = 512) -> torch.Tensor:
    ae.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size].unsqueeze(1).to(device)
            _, z = ae(batch)
            latents.append(z.cpu())
    return torch.cat(latents)


def load_and_encode(data_path: Path, ae, device, test_ratio: float = 0.2, seed: int = 42):
    data = torch.load(data_path, map_location="cpu")
    frames = data["frames"]
    actions = data["actions"]
    bounds = data["episode_boundaries"].tolist()

    print(f"[IDM] {len(frames)} frames, {len(bounds)-1} episodes, encoding...")
    z = encode_frames(frames, ae, device)

    n_eps = len(bounds) - 1
    rng = np.random.default_rng(seed)
    idx = np.arange(n_eps)
    rng.shuffle(idx)
    n_test = int(np.ceil(test_ratio * n_eps))
    test_eps = set(idx[:n_test].tolist())

    train_z, train_a, test_z, test_a = [], [], [], []
    for ep in range(n_eps):
        s, e = bounds[ep], bounds[ep + 1]
        if ep in test_eps:
            test_z.append(z[s:e])
            test_a.append(actions[s:e])
        else:
            train_z.append(z[s:e])
            train_a.append(actions[s:e])

    train_z, train_a = torch.cat(train_z), torch.cat(train_a)
    test_z, test_a = torch.cat(test_z), torch.cat(test_a)
    print(f"[IDM] train {len(train_z)}, test {len(test_z)}")
    return train_z, train_a, test_z, test_a
