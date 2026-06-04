# python rename_to_pattern.py
from pathlib import Path
from collections import defaultdict

root = Path("data/raw_videos")
for class_dir in (root / "val").iterdir():
    if not class_dir.is_dir(): 
        continue
    counter = 1
    for f in sorted(class_dir.glob("*")):
        if not f.is_file(): 
            continue
        new_name = f"{class_dir.name}_{counter:04d}.mp4"
        f.rename(class_dir / new_name)
        counter += 1