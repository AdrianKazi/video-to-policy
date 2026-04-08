import cv2
import os
from config.paths import VIDEO_DIR, FRAMES_DIR as SAVE_DIR


def extract_frames():
    print("VIDEO_DIR:", VIDEO_DIR)

    if not os.path.exists(VIDEO_DIR):
        raise Exception(f"VIDEO_DIR does not exist: {VIDEO_DIR}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    if len(os.listdir(SAVE_DIR)) > 0:
        print('[SKIP] Frames already exist — skipping extraction')
        return

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

    if len(video_files) == 0:
        raise Exception(f"No .mp4 files found in {VIDEO_DIR}")

    for vid_idx, video_name in enumerate(video_files):
        print(f"[PROCESSING] {video_name}")

        video_path = os.path.join(VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)

        episode_dir = os.path.join(SAVE_DIR, str(vid_idx))
        os.makedirs(episode_dir, exist_ok=True)

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (84, 84))

            save_path = os.path.join(episode_dir, f"{frame_id}.png")
            cv2.imwrite(save_path, frame)

            frame_id += 1

        cap.release()

        print(f"[DONE] episode {vid_idx} → {frame_id} frames")

    print("ALL DONE")


if __name__ == "__main__":
    extract_frames()