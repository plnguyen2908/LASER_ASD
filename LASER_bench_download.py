import argparse
import os
import subprocess
import shutil
import zipfile
import pandas as pd
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help="Root directory where LASER-bench will be saved",
)
args = parser.parse_args()

SAVE_ROOT = Path(args.save_path)
NOISE_DIRS = ["noise_0", "noise_1"]   # HF: noise_0/, noise_1/
SPLITS = ["val"]                      # extend to ["train", "val"] if you add train later

def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    subprocess.run(["wget", url, "-O", str(out_path)], check=True)

def download_and_extract(blob_url: str, zip_path: Path):
    """Download a .zip from HF and extract it; delete the zip afterwards."""
    raw_url = blob_url.replace("/blob/", "/resolve/")
    download(raw_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=zip_path.parent)
    zip_path.unlink()

start = time.time()

for noise_dir in NOISE_DIRS:
    print(f"=== Processing {noise_dir} ===")

    local_noise_root = SAVE_ROOT / noise_dir
    (local_noise_root / "csv").mkdir(parents=True, exist_ok=True)

    hf_base = (
        "https://huggingface.co/datasets/"
        f"plnguyen2908/LASER-bench/blob/main/{noise_dir}"
    )

    for split in SPLITS:
        # 1) Download merged CSV (already provided in LASER-bench)
        csv_blob = f"{hf_base}/csv/{split}_orig.csv"
        local_csv = local_noise_root / "csv" / f"{split}_orig.csv"
        print(f"[{noise_dir}] downloading {split} CSV -> {local_csv}")
        download(csv_blob.replace("/blob/", "/resolve/"), local_csv)

        # 2) Read video IDs from CSV
        df = pd.read_csv(local_csv)
        if "video_id" not in df.columns:
            raise ValueError(f"'video_id' column not found in {local_csv}")
        video_ids = sorted(df["video_id"].unique())
        print(f"[{noise_dir}] found {len(video_ids)} unique videos in {split}")

        # 3) Download & unzip clips for each video_id
        for vid in video_ids:
            # videos
            vid_zip_blob = f"{hf_base}/clips_videos/{split}/{vid}.zip"
            vid_zip_local = (
                local_noise_root / "clips_videos" / split / f"{vid}.zip"
            )
            print(f"  [videos] {vid}")
            download_and_extract(vid_zip_blob, vid_zip_local)

            # audios
            aud_zip_blob = f"{hf_base}/clips_audios/{split}/{vid}.zip"
            aud_zip_local = (
                local_noise_root / "clips_audios" / split / f"{vid}.zip"
            )
            print(f"  [audios] {vid}")
            download_and_extract(aud_zip_blob, aud_zip_local)

end = time.time()
print(f"Done. Total time: {end - start:.1f} seconds")
