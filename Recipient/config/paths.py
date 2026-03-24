import os

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
RECIPIENT_DIR = os.path.join(ROOT_DIR, "Recipient")
DONOR_DIR = os.path.join(ROOT_DIR, "Donor")

VIDEO_DIR = os.path.join(DONOR_DIR, "videos")
FRAMES_DIR = os.path.join(RECIPIENT_DIR, "data", "frames")
DATASETS_DIR = os.path.join(RECIPIENT_DIR, "data", "datasets")
RUNS_DIR = os.path.join(RECIPIENT_DIR, "runs")