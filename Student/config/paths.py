import os

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
STUDENT_DIR = os.path.join(ROOT_DIR, "Student")
TEACHER_DIR = os.path.join(ROOT_DIR, "Teacher")

VIDEO_DIR = os.path.join(TEACHER_DIR, "videos")
FRAMES_DIR = os.path.join(STUDENT_DIR, "data", "frames")
DATASETS_DIR = os.path.join(STUDENT_DIR, "data", "datasets")
RUNS_DIR = os.path.join(STUDENT_DIR, "runs")