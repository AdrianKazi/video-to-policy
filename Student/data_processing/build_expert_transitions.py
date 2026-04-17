import os
import torch
import numpy as np
from PIL import Image

from config.paths import FRAMES_DIR, DATASETS_DIR
from models.autoencoder import AutoEncoder


def find_latest_autoencoder(runs_dir):
    from config.paths import RUNS_DIR
    ae_runs = [d for d in os.listdir(RUNS_DIR) if d.startswith("autoencoder_")]
    if not ae_runs:
        raise FileNotFoundError(f"No autoencoder runs found in {RUNS_DIR}")
    ae_runs.sort()
    return os.path.join(RUNS_DIR, ae_runs[-1], "model.pth")


def load_encoder(model_path=None):
    if model_path is None:
        model_path = find_latest_autoencoder(None)

    print(f"[ENCODER] Loading from {model_path}")

    ae = AutoEncoder(z_dim=64)
    ae.load_state_dict(torch.load(model_path, weights_only=True))
    ae.eval()
    return ae


def encode_episode_frames(encoder, episode_dir, device="cpu"):
    frame_files = sorted(
        [f for f in os.listdir(episode_dir) if f.endswith(".png")],
        key=lambda x: int(x.split(".")[0])
    )

    latents = []

    with torch.no_grad():
        for fname in frame_files:
            img = Image.open(os.path.join(episode_dir, fname)).convert("L").resize((84, 84))
            x = torch.from_numpy(np.array(img)).float() / 255.0
            x = x.unsqueeze(0).unsqueeze(0).to(device)

            z = encoder.encoder(x)  
            latents.append(z.squeeze(0))

    return torch.stack(latents)  # (T, z_dim)


def build_expert_transitions(autoencoder_path=None, stride=1):
    encoder = load_encoder(autoencoder_path)

    episodes = sorted(
        [d for d in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, d))],
        key=lambda x: int(x)
    )

    all_transitions = []

    for ep in episodes:
        ep_dir = os.path.join(FRAMES_DIR, ep)
        latents = encode_episode_frames(encoder, ep_dir)

        for t in range(0, len(latents) - stride, stride):
            z_t = latents[t]
            z_t1 = latents[t + stride]
            all_transitions.append(torch.stack([z_t, z_t1])) 

        print(f"[TRANSITIONS] Episode {ep:>3} → {len(latents)} frames, "
              f"{max(0, (len(latents) - stride) // stride)} transitions")

    transitions = torch.stack(all_transitions) 

    # Save transitionss
    os.makedirs(DATASETS_DIR, exist_ok=True)
    save_path = os.path.join(DATASETS_DIR, "expert_transitions.pt")
    torch.save(transitions, save_path)

    print(f"\n[SAVED] {transitions.shape[0]} expert transitions → {save_path}")
    print(f"[SHAPE] {transitions.shape}")

    return transitions


if __name__ == "__main__":
    build_expert_transitions()
