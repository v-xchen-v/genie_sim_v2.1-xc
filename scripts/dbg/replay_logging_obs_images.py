import os
import pickle
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm

# -------- CONFIG --------
# task_name = "iros_make_a_sandwich"
# task_name = "iros_clear_table_in_the_restaurant"
task_name = "iros_restock_supermarket_items"
# task_name = "iros_stamp_the_seal"
# task_name = "iros_pack_moving_objects_from_conveyor"
# task_name = "iros_clear_the_countertop_waste"
EXP_ID="port_12020"
BASE_LOG_DIR = (
    f"/root/workspace/main/action_logs/{EXP_ID}/{task_name}"
)
VIDEO_ROOT = BASE_LOG_DIR.replace("action_logs", "replay_logs_video")  # you can also set manually
VIDEO_FILENAME = "stacked_view_video.mp4"
VIDEO_OUTPUT = f"/root/workspace/main/action_logs/{EXP_ID}/{task_name}/stacked_view_video.mp4"
N = 100  # Use first N files sorted by timestamp
REMOVE_PKL_AFTER_VIDEO=True
from datetime import datetime

def find_all_iter_dirs(base_dir):
    return sorted(
        [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("iter_")
        ],
        key=lambda d: int(os.path.basename(d).split("_")[1])
    )

# -------- UTILITY FUNCTIONS --------
def extract_timestamp(filename):
    try:
        parts = filename.replace(".pkl", "").split("_")
        ts_str = f"{parts[1]}_{parts[2]}"
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except (IndexError, ValueError) as e:
        print(f"⚠️ Skipping file with bad timestamp: {filename} ({e})")
        return datetime.min  # Push to start for safe sorting


def load_and_stack_images(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Extract RGB images
    img_left = data["images"]["head_left"]
    img_head = data["images"]["cam_top"]
    img_right = data["images"]["head_right"]

    # Normalize and convert to uint8 if needed
    def to_uint8(img):
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # CHW to HWC
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return img

    img_left = to_uint8(img_left)
    img_head = to_uint8(img_head)
    img_right = to_uint8(img_right)

    # Concatenate horizontally
    combined = np.concatenate((img_left, img_head, img_right), axis=1)
    return combined


# -------- MAIN PIPELINE --------
def collect_sorted_pkl_files(log_dir, n=None):
    pkl_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.startswith("observations_") and f.endswith(".pkl")
    ]
    pkl_files.sort(key=lambda x: extract_timestamp(os.path.basename(x)))
    if n is None:
        n = len(pkl_files)
    elif n > len(pkl_files):
        n = len(pkl_files)    
    return pkl_files[:n]


def make_video_from_pkl(pkl_files, output_path):
    frames = []
    for path in tqdm(pkl_files, desc="Processing frames"):
        try:
            frame = load_and_stack_images(path)
            frames.append(frame)
        except Exception as e:
            print(f"❌ Failed to process {path}: {e}")
    print(f"Total frames: {len(frames)}")
    if frames:
        print(f"Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
    # Write to video
    if frames:
        imageio.mimsave(output_path, frames, fps=1)
        print(f"✅ Video saved to: {output_path}")
    else:
        print("⚠️ No valid frames to save.")


# -------- RUN --------
if __name__ == "__main__":
    iter_dirs = find_all_iter_dirs(BASE_LOG_DIR)
    if not iter_dirs:
        print(f"No iter_x folders found in {BASE_LOG_DIR}")
        exit(1)

    for iter_dir in iter_dirs:
        # Mirror output dir path
        video_output_dir = iter_dir.replace(BASE_LOG_DIR, VIDEO_ROOT)
        os.makedirs(video_output_dir, exist_ok=True)
        output_path = os.path.join(video_output_dir, VIDEO_FILENAME)
        if os.path.exists(output_path):
            print(f"Skipping {iter_dir} (video already exists)")
            continue

        print(f"Processing {iter_dir}")
        pkl_files = collect_sorted_pkl_files(iter_dir, N)
        if pkl_files:
            make_video_from_pkl(pkl_files, output_path)
        else:
            print("No observation files found.")
