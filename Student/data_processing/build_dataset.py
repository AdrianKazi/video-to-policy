import os
import torch
import numpy as np
from PIL import Image
import random
from config.paths import FRAMES_DIR, DATASETS_DIR

SAVE_ROOT = DATASETS_DIR
TOTAL_FRAC = 0.2
STRIDE = 1


def build_sequences(input_dir, episodes):
    sequences = []

    for episode in episodes:
        episode_path = os.path.join(input_dir, episode)

        if not os.path.isdir(episode_path):
            continue

        frames = sorted(os.listdir(episode_path), key=lambda x: int(x.split('.')[0]))

        seq = []

        for i in range(0, len(frames), STRIDE):
            f = os.path.join(episode_path, frames[i])

            img = Image.open(f).convert('L').resize((84, 84))
            x = torch.from_numpy(np.array(img)).float() / 255

            seq.append(x.unsqueeze(0))  # (1,84,84)

        if len(seq) > 1:
            seq = torch.stack(seq)  # (T,1,84,84)
            sequences.append(seq)

        print(f"[DATASET] episode {episode:>3} → len={len(seq)}")

    return sequences


def build_dataset():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    episodes = sorted(os.listdir(FRAMES_DIR))
    random.shuffle(episodes)

    episodes = episodes[:int(len(episodes) * TOTAL_FRAC)]

    split_idx = int(0.8 * len(episodes))
    train_eps = episodes[:split_idx]
    test_eps = episodes[split_idx:]

    print('\n[BUILD TRAIN]\n')
    train_sequences = build_sequences(FRAMES_DIR, train_eps)

    print('\n[BUILD TEST]\n')
    test_sequences = build_sequences(FRAMES_DIR, test_eps)

    torch.save(train_sequences, os.path.join(SAVE_ROOT, 'train.pt'))
    torch.save(test_sequences, os.path.join(SAVE_ROOT, 'test.pt'))

    print('\n[SAVED]')
    print('Train:', len(train_sequences))
    print('Test: ', len(test_sequences))


if __name__ == '__main__':
    build_dataset()