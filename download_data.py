"""Download the SisFall dataset from Kaggle into ./data/sisfall."""
import shutil
from pathlib import Path

import kagglehub

# kagglehub saves to a hidden cache folder (~/.cache/kagglehub)
cache_path = kagglehub.dataset_download("nvnikhil0001/sisfall-enhanced")
print("Downloaded to cache:", cache_path)

# Copy it into this project so it's easy to find
dest = Path(__file__).parent / "data" / "sisfall"
shutil.copytree(cache_path, dest, dirs_exist_ok=True)
print("Copied to:", dest.resolve())
