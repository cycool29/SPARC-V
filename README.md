# SCN: Dilated Silhouette Convolutional Network

This repository now uses a single standardized data layout for all data stages.

## Standard Data Structure

```text
data/
	raw_videos/
		train/
			Fight/
			NonFight/
		val/
			Fight/
			NonFight/
	videos/
		train/<Class>/*.mp4
		val/<Class>/*.mp4
	silhouettes/
		train/<Class>/<video_name>/frame_*.npy
		val/<Class>/<video_name>/frame_*.npy
	point_clouds/
		train/<Class>/<video_name>.npz
		val/<Class>/<video_name>.npz
```

## Setup

```bash
pip install -r requirements.txt
```

## Training Workflow (Preprocessing Included)

```bash
# 1) Preprocess raw videos (keeps train/val/class folders)
python preprocess_videos.py --input_dir data/raw_videos --output_dir data/videos --fps 10 --width 640 --height 480

# 2) Extract 128-point silhouette boundaries
python extract_silhouettes.py --video_dir data/videos --output_dir data/silhouettes --boundary_points 128 --model yolov8s-seg.pt --device cuda

# Optional: preview silhouettes on one clip before running a full extraction
python silhouette_demo.py --video data/videos/train/Fight/sample.mp4 --device cuda --show

# 3) Build 60-frame stacked silhouette point clouds
python build_point_clouds.py --silhouette_dir data/silhouettes --output_dir data/point_clouds --n_points 4096 --temporal_window 60 --boundary_points 128

# 4) Train SCN (train/val split files are auto-generated from data/point_clouds/train and data/point_clouds/val)
python train.py --dataset HMDB --data_dir data/point_clouds --n_classes 2 --epochs 100 --batch_size 64 --output_dir checkpoints/scn

# 5) Evaluate on validation split
python evaluate.py --data_dir data/point_clouds --split checkpoints/scn/splits/test_split1.txt --checkpoint checkpoints/scn/best_model.pth --n_classes 2
```

## Todo
- [ ] Test with different YOLOv8 
- [ ] Silhouttes preview/demo-able extracted videos
