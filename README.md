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

# 3) Build 60-frame stacked silhouette point clouds
python build_point_clouds.py --silhouette_dir data/silhouettes --output_dir data/point_clouds --n_points 4096 --temporal_window 60 --boundary_points 128

# 4) Train SCN (train/val split files are auto-generated from data/point_clouds/train and data/point_clouds/val)
python train.py --dataset HMDB --data_dir data/point_clouds --n_classes 2 --epochs 100 --batch_size 64 --output_dir checkpoints/scn

# 5) Evaluate on validation split
python evaluate.py --data_dir data/point_clouds --split checkpoints/scn/splits/test_split1.txt --checkpoint checkpoints/scn/best_model.pth --n_classes 2
```

## Technical Methodology

This project implements a silhouette-based spatio-temporal action recognition pipeline for violence detection. The core idea is to convert each video into a compact 3D point-cloud representation built from human silhouette boundaries, then classify that point cloud with the Dilated Silhouette Convolutional Network (SCN).

### 1. Video preprocessing

Raw videos are first normalized with `ffmpeg` so every clip has a consistent format before any vision model touches it.

The preprocessing stage does four things:
- Re-encodes clips into H.264 MP4 for consistent decoding.
- Resamples frames to a fixed FPS, which stabilizes temporal spacing between clips.
- Scales each video into a fixed spatial envelope while preserving aspect ratio.
- Pads the frame to the requested output resolution so all clips share the same geometry.

This matters because the downstream silhouette extraction and point-cloud construction assume a stable frame rate and comparable image geometry across the dataset.

### 2. Silhouette boundary extraction

The first semantic step is to isolate people in each frame and turn the mask into a boundary curve.

The extractor supports two paths:
- YOLOv8 instance segmentation, which is the main path.
- OpenCV MOG2 background subtraction, which is a lightweight fallback for static-camera footage.

For each processed frame, the model produces person masks. Those masks are merged, resized back to the original frame shape if necessary, and converted to contours using `cv2.findContours`. The result is not a dense segmentation mask but an ordered silhouette boundary.

Each contour is then uniformly resampled to a fixed number of boundary points, typically 128. Uniform resampling is important because it converts variable-length contours into a fixed-size geometric descriptor and keeps the boundary sampling invariant to contour length.

### 3. Temporal stacking into a 3D point cloud

After silhouettes are extracted frame-by-frame, the project turns a video into a 3D spatio-temporal point cloud.

Each frame contributes a set of 2D boundary samples `(x, y)`. The frame index is used as the third coordinate `z`, normalized into `[0, 1]`. This creates points of the form `(x, y, z)` where:
- `x, y` describe the silhouette shape in the image plane.
- `z` describes when that boundary point occurs in the clip.

Before stacking, each frame boundary is centered around its own centroid so the representation is less sensitive to global translation inside the frame. Then all points from the selected temporal window are concatenated into one cloud, normalized in the XY plane, and downsampled with Farthest Point Sampling to a fixed size of 4096 points.

The temporal window is set to 60 frames by default. When a video has more than 60 extracted silhouette frames, the central window is selected so the cloud captures the most informative middle portion of the action rather than only the start or end of the clip.

### 4. Slow-to-Fast temporal decomposition

The project does not rely on a single temporal scale. Instead, it builds three point-cloud views of the same video:
- Slow: all frames.
- Faster: every 2nd frame.
- Fastest: every 3rd frame.

This multi-scale design is intended to capture different kinds of motion. Slow sampling preserves detailed temporal structure, while faster and fastest sampling reduce temporal redundancy and emphasize coarser motion changes.

In implementation, the three views are produced independently and stored inside each `.npz` file as separate point sets with their own density coefficients.

### 5. Density-aware point representation

Each downsampled point cloud is paired with a density coefficient computed by a Gaussian KDE-like estimate.

The intuition is that dense regions of the cloud should contribute differently from sparse regions. The project therefore stores an inverse density value per point, which is later consumed by the network as an explicit weighting signal.

This gives the model a geometric notion of local support: points from crowded contour regions and points from sparse regions do not look identical to the network, even if their coordinates are similar.

### 6. Dilated Silhouette Convolution

The main SCN block is a local neighborhood operator over point clouds.

For each centroid point, the network finds its `k` nearest neighbors in the same cloud. The implementation uses kNN over the 3D point set, where the temporal coordinate `z` is part of the distance metric. The neighborhood is then processed by two branches:
- A coordinate branch that consumes relative coordinates plus neighbor features.
- A density branch that consumes the neighbor density coefficients.

Both branches are built from 1x1 convolutions and ReLU/BatchNorm layers. Their outputs are fused element-wise, then pooled across the neighborhood. This acts like a learned local geometric filter that is aware of both shape and density.

The term "dilated" refers to the fact that the z-coordinate is scaled by a dilation factor before local aggregation. A larger dilation expands the receptive field along the temporal axis, so the same convolution can model both short-range and longer-range motion patterns.

### 7. Hierarchical SCN encoder

The full encoder is hierarchical.

It first performs Farthest Point Sampling to reduce the cloud to 512 centroids, then applies the dilated convolution blocks. It then samples again down to 128 centroids and repeats the local processing. Finally, a 1D MLP projects the features to a 1024-dimensional global embedding and global max pooling collapses the point dimension.

This hierarchy gives the model two benefits:
- It reduces compute by shrinking the number of active points.
- It lets the network learn both local contour structure and higher-level spatio-temporal interactions.

### 8. Slow-to-Fast classifier head

The model uses three parallel encoders, one for each temporal scale. Their 1024-dimensional outputs are concatenated into a 3072-dimensional feature vector and passed through a fully connected classifier with dropout and BatchNorm.

This design makes the final prediction depend on multiple views of the same event rather than a single fixed sampling rate. In practice, that gives the model more robustness when an action is brief, partially occluded, or not evenly distributed across the clip.

### 9. Training objective and optimization

Training uses standard multi-class cross-entropy classification, even though the current use case is binary violence detection. The label space is set by the class folders and the `n_classes` argument.

Optimization details:
- Optimizer: Adam.
- Initial learning rate: 0.001.
- Weight decay: 0.0001.
- Gradient clipping: max norm 10.
- Learning rate schedule: fixed phases that reduce the learning rate mid-training and near the end.

The training loop also uses micro-batching inside each loader batch. This keeps the effective batch size stable while reducing peak GPU memory.

### 10. Data organization and split logic

The project now assumes a standardized split-aware directory structure:
- `train` contains the optimization set.
- `val` contains the held-out evaluation set.
- The split generator creates `train_split1.txt` from `train` and `test_split1.txt` from `val`.

That means evaluation is not mixing in training samples unless the wrong split file is passed manually.

## Novelty

The main novelty of this project is not just that it detects violence, but that it does so using a geometry-first spatio-temporal representation rather than relying on raw RGB or dense frame-level appearance cues.

### What is different from typical CCTV violence models

Most CCTV violence detectors are built on one of these patterns:
- 2D CNNs over RGB frames.
- 3D CNNs or two-stream video models.
- Optical-flow-based methods.
- Heavy transformer-style video encoders.

This project instead compresses each clip into a structured silhouette point cloud and classifies motion through local geometric neighborhoods. That changes the problem from appearance-heavy video understanding to shape-aware motion analysis.

### Advantages

- Better focus on human motion: the model works on silhouette geometry, so it is less distracted by background clutter, lighting changes, camera color balance, or scene texture.
- Stronger privacy profile: the pipeline does not need to preserve identifiable RGB detail after preprocessing, which is helpful for CCTV-style deployments.
- Lower input redundancy: the contour-based representation is much smaller than full-frame RGB streams, so the model processes less irrelevant information.
- Multi-scale temporal modeling: the slow/faster/fastest streams let the classifier inspect the same event at different temporal resolutions.
- Temporal dilation support: the convolutional blocks can expand their temporal receptive field without changing the whole architecture.
- Density-aware local reasoning: the network explicitly models how concentrated or sparse the silhouette support is in each region of the point cloud.
- More robust to static backgrounds: because the representation is based on extracted human shape, it is less sensitive to the fixed scene content common in CCTV environments.

### Practical value for CCTV settings

For CCTV violence detection, the hardest part is often not the classifier itself but the fact that the video stream contains a lot of irrelevant scene detail. By converting the stream into a person-centric silhouette cloud, this project aims to isolate the interaction signal that actually matters: body pose, motion timing, and multi-person proximity over time.

That makes the pipeline a better fit for settings where privacy, compute efficiency, and background robustness matter more than raw RGB fidelity.

## Files
- `preprocess_videos.py`     — ffmpeg batch preprocessing (fps/resize/pad, split/class-preserving)
- `extract_silhouettes.py`   — YOLOv8/MOG2 silhouette boundary extraction
- `build_point_clouds.py`    — FPS resampling + stacked 3D point cloud construction
- `dataset.py`               — PyTorch dataset and split-file generation
- `model.py`                 — Full SCN model (dilated conv + Slow-to-Fast)
- `train.py`                 — Training loop with Adam optimizer
- `evaluate.py`              — Evaluation on test/validation splits
- `utils.py`                 — FPS, KDE density estimation, helpers
- `live_monitor.py`          — RTSP/LAN tracker, SCN inference, and local alerts
