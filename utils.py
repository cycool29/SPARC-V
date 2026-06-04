"""
utils.py — Utility functions for SCN training pipeline.

Implements:
  - Farthest Point Sampling (FPS) as described in Section 3.1
  - KDE-based point density estimation for density coefficients (Section 3.2.1)
  - Point cloud normalization helpers
"""

import numpy as np
from typing import Any, TYPE_CHECKING

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Provide types for static checkers without requiring torch at runtime
if TYPE_CHECKING:
    import torch  # pragma: no cover


# ---------------------------------------------------------------------------
# Farthest Point Sampling (FPS)
# Paper Section 3.1: "the Farthest Point Sampling (FPS) algorithm is adopted
# for downsampling. Compared with random sampling, FPS can get better
# coverage of the entire point set."
# ---------------------------------------------------------------------------

def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Downsample a point cloud to n_samples points using FPS.

    Args:
        points:    (N, 3) array of 3D points
        n_samples: target number of points m

    Returns:
        sampled:   (m, 3) array of selected points
    """
    N = points.shape[0]
    if N <= n_samples:
        # Pad by repeating if fewer points than requested
        repeat = int(np.ceil(n_samples / N))
        points = np.tile(points, (repeat, 1))[:n_samples]
        return points

    # Step 1: initialise selected set B with a random starting point
    selected_idx = np.zeros(n_samples, dtype=int)
    selected_idx[0] = np.random.randint(0, N)

    # Distance from each point to the nearest selected point
    distances = np.full(N, np.inf)

    for i in range(1, n_samples):
        last = points[selected_idx[i - 1]]
        # Update minimum distances
        d = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, d)
        # Select the farthest point
        selected_idx[i] = np.argmax(distances)

    return points[selected_idx]


def farthest_point_sampling_torch(points: 'torch.Tensor', n_samples: int) -> 'torch.Tensor':  # type: ignore
    """
    GPU-friendly FPS for batched point clouds.

    Args:
        points:    (B, N, 3) tensor
        n_samples: target m

    Returns:
        sampled:   (B, m, 3) tensor
    """
    if torch is None:
        raise ImportError("PyTorch is required for farthest_point_sampling_torch")

    B, N, C = points.shape
    device = points.device

    selected_idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    selected_idx[:, 0] = torch.randint(0, N, (B,), device=device)

    distances = torch.full((B, N), float('inf'), device=device)

    for i in range(1, n_samples):
        last_pts = points[torch.arange(B), selected_idx[:, i - 1], :]  # (B, 3)
        d = torch.sum((points - last_pts.unsqueeze(1)) ** 2, dim=-1)   # (B, N)
        distances = torch.minimum(distances, d)
        selected_idx[:, i] = torch.argmax(distances, dim=-1)

    # Gather selected points and ensure memory contiguity for subsequent processing stages
    idx_expanded = selected_idx.unsqueeze(-1).expand(B, n_samples, 3)
    sampled = torch.gather(points, 1, idx_expanded).contiguous()
    return sampled


# ---------------------------------------------------------------------------
# K-Nearest Neighbours (for local neighbourhood extraction)
# ---------------------------------------------------------------------------

def knn_query(query_pts: 'torch.Tensor', support_pts: 'torch.Tensor', k: int, chunk_size: int = 128) -> 'torch.Tensor':  # type: ignore
    """
    For each query point, find the k nearest neighbours in support_pts.

    Args:
        query_pts:   (B, M, 3)
        support_pts: (B, N, 3)
        k:           number of neighbours

    Returns:
        idx: (B, M, k) indices into support_pts
    """
    B, M, _ = query_pts.shape
    indices = []

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = query_pts[:, start:end, :]  # (B, chunk_size, 3)
        
        # torch.cdist bounds peak memory; squared to obtain distance squared values
        dist2 = torch.cdist(chunk, support_pts, p=2) ** 2
        _, idx = torch.topk(dist2, k, dim=-1, largest=False)
        indices.append(idx)

    # Enforce sequence contiguity to prevent indexing slows down the pipeline
    return torch.cat(indices, dim=1).contiguous()


# ---------------------------------------------------------------------------
# Kernel Density Estimation (KDE) for density coefficients
# Paper Section 3.2.1: "we first calculate the density coefficient of each
# point in an offline manner" using the reciprocal of KDE value.
# ---------------------------------------------------------------------------

def compute_density_coefficients(points: np.ndarray, bandwidth: float = 0.05) -> np.ndarray:
    """
    Estimate point density via a Gaussian KDE for each point and return
    the reciprocal (used to down-weight dense regions).

    Args:
        points:    (N, 3) point cloud
        bandwidth: kernel bandwidth h

    Returns:
        density_inv: (N,) reciprocal density coefficients
    """
    N = points.shape[0]
    # Pairwise squared distances
    diff = points[:, None, :] - points[None, :, :]  # (N, N, 3)
    dist2 = np.sum(diff ** 2, axis=-1)               # (N, N)
    # Gaussian kernel: sum over all neighbours
    density = np.sum(np.exp(-dist2 / (2 * bandwidth ** 2)), axis=1) / N
    density = np.maximum(density, 1e-6)              # avoid division by zero
    return 1.0 / density                             # reciprocal


def compute_density_coefficients_torch(
    points: 'torch.Tensor', bandwidth: float = 0.05  # type: ignore
) -> 'torch.Tensor':  # type: ignore
    """
    Batched GPU version of density coefficient computation.
    Optimized to use cdist to avoid large (B, N, N, 3) tensor allocation explosions.

    Args:
        points: (B, N, 3)

    Returns:
        density_inv: (B, N)
    """
    B, N, _ = points.shape
    
    # Highly optimized matrix-multiplication based distance computation to mitigate OOMs
    dist2 = torch.cdist(points, points, p=2) ** 2                                # (B, N, N)
    density = torch.sum(torch.exp(-dist2 / (2 * bandwidth ** 2)), dim=-1) / N    # (B, N)
    density = density.clamp(min=1e-6)
    
    return (1.0 / density).contiguous()


# ---------------------------------------------------------------------------
# Point cloud normalisation
# ---------------------------------------------------------------------------

def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Centre and scale a point cloud to fit in the unit sphere.

    Args:
        points: (N, 3)

    Returns:
        points: (N, 3) normalised
    """
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points


def normalize_point_cloud_torch(points: 'torch.Tensor') -> 'torch.Tensor':  # type: ignore
    """
    Batched normalisation. points: (B, N, 3)
    """
    centroid = points.mean(dim=1, keepdim=True)
    points = points - centroid
    scale = points.norm(dim=-1).max(dim=-1, keepdim=True)[0].unsqueeze(-1) + 1e-8
    return points / scale


def resample_contour_uniform(contour: np.ndarray, n_points: int = 128) -> np.ndarray:
    """
    Uniformly resample a 2D boundary curve along arc length.

    Args:
        contour:  (N, 2) boundary points ordered along the contour
        n_points: target number of samples

    Returns:
        (n_points, 2) float32 array
    """
    contour = np.asarray(contour, dtype=np.float32)
    if contour.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(f"Expected contour shape (N, 2), got {contour.shape}")

    if contour.shape[0] == 1:
        return np.repeat(contour, n_points, axis=0)

    closed = contour
    if not np.allclose(contour[0], contour[-1]):
        closed = np.concatenate([contour, contour[:1]], axis=0)

    deltas = np.diff(closed, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-8:
        return np.repeat(contour[:1], n_points, axis=0)

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    sample_positions = np.linspace(0.0, total_length, n_points, endpoint=False)

    x = np.interp(sample_positions, cumulative, closed[:, 0])
    y = np.interp(sample_positions, cumulative, closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def build_point_cloud_from_frames(
    frame_boundaries: list,
    n_points: int = 4096,
    bandwidth: float = 0.05,
) -> tuple:
    """
    Convert a temporal sequence of 2D boundary samples into a 3D stacked point cloud.

    Each boundary is interpreted as a frame slice. The frame index is encoded
    in z, then the full cloud is normalized, downsampled with FPS, and assigned
    offline density coefficients.
    """
    all_points = []
    n_frames = len(frame_boundaries)

    for t, boundary in enumerate(frame_boundaries):
        pts_2d = np.asarray(boundary, dtype=np.float32)
        if pts_2d.size == 0:
            continue
        if pts_2d.ndim != 2 or pts_2d.shape[1] != 2:
            raise ValueError(f"Expected frame boundary shape (N, 2), got {pts_2d.shape}")

        cog = pts_2d.mean(axis=0)
        pts_2d = pts_2d - cog
        z = np.full((pts_2d.shape[0], 1), t / max(n_frames - 1, 1), dtype=np.float32)
        all_points.append(np.concatenate([pts_2d, z], axis=1))

    if not all_points:
        return (
            np.random.randn(n_points, 3).astype(np.float32),
            np.ones(n_points, dtype=np.float32),
        )

    raw_cloud = np.concatenate(all_points, axis=0).astype(np.float32)
    xy_max = np.abs(raw_cloud[:, :2]).max()
    if xy_max > 0:
        raw_cloud[:, :2] /= xy_max

    point_cloud = farthest_point_sampling(raw_cloud, n_points).astype(np.float32)
    density_inv = compute_density_coefficients(point_cloud, bandwidth=bandwidth).astype(np.float32)
    return point_cloud, density_inv


# ---------------------------------------------------------------------------
# Temporal sub-sampling for Slow-to-Fast (Section 3.2.3)
# ---------------------------------------------------------------------------

def temporal_subsample(points: np.ndarray, frame_ids: np.ndarray, stride: int) -> np.ndarray:
    """
    Keep only points whose frame index (z coordinate mapped to frame_ids)
    falls on a stride boundary, mimicking S_faster / S_fastest sampling.

    Args:
        points:    (N, 3) — z axis encodes frame index
        frame_ids: sorted array of unique frame indices present
        stride:    1=slow (all), 2=faster (every 2nd), 3=fastest (every 3rd)

    Returns:
        subset: (M, 3) point cloud
    """
    if stride == 1:
        return points
    keep_frames = set(frame_ids[::stride].tolist())
    # Map z back to frame index — assumes z was normalised during construction
    # so we use closest-frame matching
    unique_z = np.unique(points[:, 2])
    z_to_frame = {z: frame_ids[i] for i, z in enumerate(unique_z)}
    mask = np.array([
        z_to_frame.get(p[2], -1) in keep_frames for p in points
    ])
    subset = points[mask]
    # Ensure minimum points
    if subset.shape[0] < 64:
        subset = points
    return subset