import os
from data_processing.extract_frames import extract_frames
from data_processing.build_dataset import build_dataset
from data_processing.load_dataset import DownloadVideoDataset
from training.train_autoencoder import train_autoencoder
from models.autoencoder import AutoEncoder
from config.paths import DATASETS_DIR


def main():

    # =========================
    # 1. Extract frames
    # =========================
    print("\n[STEP 1] Extract frames\n")
    extract_frames()

    # =========================
    # 2. Build dataset
    # =========================
    print("\n[STEP 2] Build dataset\n")
    build_dataset()

    # =========================
    # 3. Load dataset
    # =========================
    print("\n[STEP 3] Load dataset\n")
    train_path = os.path.join(DATASETS_DIR, "train.pt")
    train_dataset = DownloadVideoDataset(train_path)

    # =========================
    # 4. Train AE
    # =========================
    print("\n[STEP 4] Train AutoEncoder\n")
    model, losses = train_autoencoder(train_dataset, AutoEncoder)


if __name__ == "__main__":
    main()