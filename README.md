# SCN: Dilated Silhouette Convolutional Network
## Training Workflow (from Hua et al., 2021)

### Setup
```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn tqdm tensorboard
pip install ultralytics opencv-python
```

### Quick Start
```bash
# 1. Extract 128-point silhouette boundaries from raw videos
python extract_silhouettes.py --video_dir data/HMDB/videos --output_dir data/HMDB/silhouettes --boundary_points 128

# 2. Build 60-frame stacked silhouette point clouds
python build_point_clouds.py --silhouette_dir data/HMDB/silhouettes --output_dir data/HMDB/point_clouds --n_points 4096 --temporal_window 60 --boundary_points 128

# 3. Train SCN
python train.py --dataset HMDB --data_dir data/HMDB/point_clouds --epochs 100 --batch_size 64

# 4. Evaluate
python evaluate.py --data_dir data/HMDB/point_clouds --split splits/test_split1.txt --checkpoint checkpoints/best_model.pth --n_classes 51

# 5. Run local RTSP/LAN monitoring
python live_monitor.py --source rtsp://camera/stream --checkpoint checkpoints/scn/best_model.pth --n_classes 2 --positive_class 1
```

### Files
- `extract_silhouettes.py`  — YOLOv8/MOG2 silhouette boundary extraction
- `build_point_clouds.py`   — FPS resampling + stacked 3D point cloud construction
- `dataset.py`              — PyTorch Dataset classes for HMDB/JHMDB/UCF101
- `model.py`                — Full SCN model (dilated conv + Slow-to-Fast)
- `train.py`                — Training loop with Adam optimizer
- `evaluate.py`             — Evaluation on test splits
- `utils.py`                — FPS, KDE density estimation, helpers
- `live_monitor.py`         — RTSP/LAN tracker, SCN inference, and local alerts
