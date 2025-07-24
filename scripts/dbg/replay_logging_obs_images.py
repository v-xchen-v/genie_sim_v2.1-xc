import os
import pickle
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm

# -------- CONFIG --------
LOG_DIR = "/home/xichen6/Documents/repos/genie_sim_v2.1/genie_sim/action_logs/iros_make_a_sandwich"
VIDEO_OUTPUT = "/home/xichen6/Documents/repos/genie_sim_v2.1/genie_sim/action_logs/iros_make_a_sandwich/stacked_view_video.mp4"
N = 20  # Use first N files sorted by timestamp


# -------- UTILITY FUNCTIONS --------
def extract_timestamp(filename):
    try:
        return filename.split("_")[1].replace(".pkl", "")
    except IndexError:
        return ""


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
def collect_sorted_pkl_files(log_dir, n):
    pkl_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.startswith("observations_") and f.endswith(".pkl")
    ]
    pkl_files.sort(key=lambda x: extract_timestamp(os.path.basename(x)))
    return pkl_files[:n]


def make_video_from_pkl(pkl_files, output_path):
    frames = []
    for path in tqdm(pkl_files, desc="Processing frames"):
        try:
            frame = load_and_stack_images(path)
            frames.append(frame)
        except Exception as e:
            print(f"❌ Failed to process {path}: {e}")

    # Write to video
    if frames:
        imageio.mimsave(output_path, frames, fps=5)
        print(f"✅ Video saved to: {output_path}")
    else:
        print("⚠️ No valid frames to save.")


# -------- RUN --------
if __name__ == "__main__":
    pkl_files = collect_sorted_pkl_files(LOG_DIR, N)
    if pkl_files:
        make_video_from_pkl(pkl_files, VIDEO_OUTPUT)
    else:
        print("No observation files found.")
