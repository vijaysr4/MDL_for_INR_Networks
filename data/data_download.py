import os
import shutil
from pathlib import Path
import kagglehub

# 1. Download Kodak dataset via kagglehub (into its cache)
src_path = kagglehub.dataset_download("sherylmehta/kodak-dataset")
print("Downloaded to:", src_path)

# 2. Target directory where you want the full dataset
dest_path = Path("kodak_all")
dest_path.mkdir(parents=True, exist_ok=True)

# 3. Copy all files and folders from the kagglehub cache to /data/kodak_all
src_path = Path(src_path)

for item in src_path.iterdir():
    target = dest_path / item.name
    if item.is_dir():
        # For Python 3.8+ you can use dirs_exist_ok=True
        shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        shutil.copy2(item, target)

print("Dataset copied to:", dest_path)
