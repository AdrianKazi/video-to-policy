
from pathlib import Path

import torch

from Autoencoder.network import AutoEncoder
from IDM.dataset import encode_frames
from IDM.network import IDMContext


def pseudo_label_expert(
    frames: torch.Tensor,
    episode_boundaries: torch.Tensor,
    ae: AutoEncoder,
    idm: IDMContext,
    context_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ae.eval()
    idm.eval()
    z = encode_frames(frames, ae, device)

    bounds = episode_boundaries.tolist()
    n_eps = len(bounds) - 1
    all_z, all_a = [], []

    with torch.no_grad():
        for ep in range(n_eps):
            s, e = bounds[ep], bounds[ep + 1]
            z_ep = z[s:e]
            if len(z_ep) <= context_len:
                continue
            # build context windows
            windows = []
            for t in range(context_len, len(z_ep)):
                windows.append(z_ep[t - context_len : t + 1])
            windows = torch.stack(windows).to(device)

            # batch predict
            actions = []
            bs = 512
            for i in range(0, len(windows), bs):
                a = idm(windows[i:i+bs])
                actions.append(a.cpu())
            actions = torch.cat(actions)

            # z for each labeled frame is z[t] where t = context_len..len-1
            all_z.append(z_ep[context_len:])
            all_a.append(actions)

    return torch.cat(all_z), torch.cat(all_a)
