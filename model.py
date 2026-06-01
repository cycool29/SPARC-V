"""
model.py — Dilated Silhouette Convolutional Network (SCN).

Implements the full architecture from Hua et al. (2021):

  Section 3.2.1 — Dilated Silhouette Convolution
    Dconv(S, W, F)_xyz = Σ S(δ)·W(δ)·F(x+δx, y+δy, T(z+δz))
    where T(·) = temporal dilation on z-axis.
    Implemented via two-branch MLP:
      Branch a: local coordinates → weight W  (1×1 conv × 2 + ReLU)
      Branch b: density values    → coeff S   (1×1 conv × 2 + ReLU)
    Output = W ⊗ S (element-wise) → global max pooling

  Section 3.2.2 — SCN Architecture
    Two DilatedSilhouetteConv layers (dilations 1 and 2 each),
    each followed by BN+ReLU. Final 1×1 conv → BN+ReLU → global max pool.

  Section 3.2.3 — Slow-to-Fast SCN
    Three parallel SCN encoders (slow / faster / fastest temporal scales).
    Their feature vectors are concatenated and fed to FC classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import farthest_point_sampling_torch, knn_query


# ---------------------------------------------------------------------------
# Building block: Dilated Silhouette Convolution (single dilation)
# ---------------------------------------------------------------------------

class DilatedSilhouetteConv(nn.Module):
    """
    One dilated silhouette convolution applied to a local neighbourhood.

    Architecture (Fig. 4 in the paper):
      Branch a (coordinates → W):
        input (B, M, K, C_in+3) → dilation → MLP [1×1 conv, ReLU, 1×1 conv] → W
      Branch b (density → S):
        input (B, M, K, 1) → dilation → MLP [1×1 conv, ReLU, 1×1 conv] → S
      Output:
        F_out = sum_k (W_k ⊗ S_k) · F_k  → global max over k → (B, M, C_out)

    Args:
        in_channels:  C_in (input feature channels)
        out_channels: C_out
        k:            number of nearest neighbours
        dilation:     temporal dilation factor for z-axis
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20, dilation: int = 1):
        super().__init__()
        self.k        = k
        self.dilation = dilation

        # Branch a: coordinate branch → weight W
        # Input dim: 3 (relative coords) + in_channels (features)
        self.coord_mlp = nn.Sequential(
            nn.Conv2d(3 + in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

        # Branch b: density branch → coefficient S
        self.density_mlp = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

        self.bn  = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def _apply_temporal_dilation(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal dilation to z-axis of point coordinates.
        T(z + δz) effectively expands the temporal receptive field.
        For dilation d: δz_dilated = δz * d  (applied to relative coords).
        """
        if self.dilation == 1:
            return pts
        dilated = pts.clone()
        dilated[..., 2] = dilated[..., 2] * self.dilation  # scale z relative offsets
        return dilated

    @staticmethod
    def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather per-batch neighbors without expanding to a full B×N×N tensor."""
        batch_indices = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1)
        batch_indices = batch_indices.expand_as(indices)
        return values[batch_indices, indices]

    def forward(
        self,
        xyz: torch.Tensor,          # (B, N, 3) — centroid coordinates
        features: torch.Tensor,     # (B, N, C_in) — per-point features
        density: torch.Tensor,      # (B, N) — density inverse coefficients
    ) -> torch.Tensor:
        """
        Returns:
            new_features: (B, N, out_channels)
        """
        B, N, _ = xyz.shape

        # 1. K-NN neighbourhood query
        knn_idx = knn_query(xyz, xyz, self.k)           # (B, N, k)

        # 2. Gather neighbour coordinates and compute local (relative) coords
        #    centroid shape: (B, N, 1, 3)
        centroid = xyz.unsqueeze(2)
        nbr_xyz  = self._gather_neighbors(xyz, knn_idx)
        local_xyz = nbr_xyz - centroid                  # (B, N, k, 3)

        # 3. Apply temporal dilation to z offsets
        local_xyz = self._apply_temporal_dilation(local_xyz)

        # 4. Gather neighbour features
        nbr_feat = self._gather_neighbors(features, knn_idx)  # (B, N, k, C_in)

        # 5. Concatenate coords + features for Branch a
        coord_input = torch.cat([local_xyz, nbr_feat], dim=-1)  # (B, N, k, 3+C_in)
        # Rearrange to (B, 3+C_in, N, k) for Conv2d
        coord_input = coord_input.permute(0, 3, 1, 2)

        W = self.coord_mlp(coord_input)                # (B, C_out, N, k)

        # 6. Gather density values for Branch b
        nbr_d   = self._gather_neighbors(density.unsqueeze(-1), knn_idx)  # (B, N, k, 1)
        density_input = nbr_d.permute(0, 3, 1, 2)    # (B, 1, N, k)

        S = self.density_mlp(density_input)            # (B, C_out, N, k)

        # 7. Eq. (4): F_out = sum_k S(k) * W(k) * F(k)  then max pool over k
        #    W ⊗ S element-wise, then max over k dimension
        fused = W * S                                  # (B, C_out, N, k)
        out   = fused.max(dim=-1)[0]                   # (B, C_out, N)

        out = self.bn(out)
        out = self.act(out)

        return out.permute(0, 2, 1)                    # (B, N, C_out)


# ---------------------------------------------------------------------------
# SCN Encoder (Fig. 5 in the paper)
# ---------------------------------------------------------------------------

class SCNEncoder(nn.Module):
    """
    Hierarchical SCN feature encoder.

    Structure (Fig. 5):
      Input: (B, N, 3) stacked silhouette point cloud
      ↓ FPS to N1 centroids
      ↓ DilatedSilhouetteConv (dilation 1) + DilatedSilhouetteConv (dilation 2)
      ↓ FPS to N2 centroids
      ↓ DilatedSilhouetteConv (dilation 1) + DilatedSilhouetteConv (dilation 2)
      ↓ 1×1 Conv + BN + ReLU
      ↓ Global max pool → feature vector of size 1024
    """

    def __init__(
        self,
        in_channels: int = 0,    # 0 means coordinates only as features
        n1: int = 512,           # centroids after first FPS (N1×3)
        n2: int = 128,           # centroids after second FPS (N2×3)
        k: int = 20,             # kNN neighbours
        base_channels: int = 128,
    ):
        super().__init__()
        self.n1 = n1
        self.n2 = n2

        # Input feature dim: 3 (xyz) if no extra features
        c0 = max(in_channels, 1)  # initial feature dim treated as 1 for density path

        # Layer 1: two dilations (1 and 2), applied on N1 centroids
        # Bootstrap the first block with raw xyz as both coordinates and features.
        self.conv1_d1 = DilatedSilhouetteConv(3,              base_channels,     k=k, dilation=1)
        self.conv1_d2 = DilatedSilhouetteConv(base_channels,  base_channels,     k=k, dilation=2)

        # Layer 2: two dilations (1 and 2), applied on N2 centroids
        self.conv2_d1 = DilatedSilhouetteConv(base_channels,  base_channels * 2, k=k, dilation=1)
        self.conv2_d2 = DilatedSilhouetteConv(base_channels * 2, base_channels * 2, k=k, dilation=2)

        # Final 1×1 conv layer (paper: mlp(256, 512, 1024))
        self.final_mlp = nn.Sequential(
            nn.Conv1d(base_channels * 2, 256,  kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        pts: torch.Tensor,       # (B, N, 3)
        density: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:
        """
        Returns:
            feature: (B, 1024) global feature vector
        """
        B, N, _ = pts.shape

        # --- Layer 1 ---
        # FPS to N1 centroids
        pts1 = farthest_point_sampling_torch(pts, self.n1)         # (B, N1, 3)
        # Interpolate density to N1 (use nearest-centroid assignment)
        density1 = self._interpolate_density(pts, density, pts1)   # (B, N1)

        # Initial features are unused by the first block; xyz is reused as a bootstrap feature.
        feat1 = torch.zeros(B, self.n1, 1, device=pts.device)      # dummy init

        # Dilated conv layer 1 — dilation 1
        feat1 = self.conv1_d1(pts1, pts1, density1)                 # (B, N1, 128)
        feat1 = self.conv1_d2(pts1, feat1, density1)                # (B, N1, 128)

        # --- Layer 2 ---
        pts2     = farthest_point_sampling_torch(pts1, self.n2)     # (B, N2, 3)
        density2 = self._interpolate_density(pts1, density1, pts2)  # (B, N2)
        feat2    = self._interpolate_features(pts1, feat1, pts2)    # (B, N2, 128)

        feat2 = self.conv2_d1(pts2, feat2, density2)                # (B, N2, 256)
        feat2 = self.conv2_d2(pts2, feat2, density2)                # (B, N2, 256)

        # --- Final 1×1 conv + global max pool ---
        x = feat2.permute(0, 2, 1)                                  # (B, 256, N2)
        x = self.final_mlp(x)                                       # (B, 1024, N2)
        x = x.max(dim=-1)[0]                                        # (B, 1024)
        return x

    @staticmethod
    def _interpolate_density(
        src_pts: torch.Tensor,  # (B, N_src, 3)
        src_den: torch.Tensor,  # (B, N_src)
        tgt_pts: torch.Tensor,  # (B, N_tgt, 3)
    ) -> torch.Tensor:
        """Nearest-neighbour interpolation of density to target points."""
        diff  = tgt_pts.unsqueeze(2) - src_pts.unsqueeze(1)  # (B, N_tgt, N_src, 3)
        dist2 = torch.sum(diff ** 2, dim=-1)                  # (B, N_tgt, N_src)
        nn_idx = dist2.argmin(dim=-1)                         # (B, N_tgt)
        tgt_den = torch.gather(src_den, 1, nn_idx)            # (B, N_tgt)
        return tgt_den

    @staticmethod
    def _interpolate_features(
        src_pts:  torch.Tensor,  # (B, N_src, 3)
        src_feat: torch.Tensor,  # (B, N_src, C)
        tgt_pts:  torch.Tensor,  # (B, N_tgt, 3)
    ) -> torch.Tensor:
        """Nearest-neighbour feature interpolation."""
        diff  = tgt_pts.unsqueeze(2) - src_pts.unsqueeze(1)  # (B, N_tgt, N_src, 3)
        dist2 = torch.sum(diff ** 2, dim=-1)                  # (B, N_tgt, N_src)
        nn_idx = dist2.argmin(dim=-1)                         # (B, N_tgt)
        C = src_feat.shape[-1]
        idx_e = nn_idx.unsqueeze(-1).expand(-1, -1, C)
        return torch.gather(src_feat, 1, idx_e)               # (B, N_tgt, C)


# ---------------------------------------------------------------------------
# Slow-to-Fast SCN (Fig. 6 in the paper)
# ---------------------------------------------------------------------------

class SlowToFastSCN(nn.Module):
    """
    Three parallel SCN encoders operating at different temporal scales:
      - SCN_slow:    all frames
      - SCN_faster:  every 2nd frame
      - SCN_fastest: every 3rd frame

    Their 1024-dim feature vectors are concatenated → 3072-dim → FC classifier.

    Paper Section 3.2.3:
      "The output features from each SCN are concatenated to a single feature
       vector and fed to the set of fully-connected layers with integrated
       dropout and ReLU layers."
    """

    def __init__(
        self,
        n_classes: int,
        n1: int = 512,
        n2: int = 128,
        k: int = 20,
        base_channels: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        encoder_kwargs = dict(n1=n1, n2=n2, k=k, base_channels=base_channels)
        self.scn_slow    = SCNEncoder(**encoder_kwargs)
        self.scn_faster  = SCNEncoder(**encoder_kwargs)
        self.scn_fastest = SCNEncoder(**encoder_kwargs)

        # Classifier: FC + dropout + ReLU (paper uses dropout 0.3 per conv layer)
        self.classifier = nn.Sequential(
            nn.Linear(3 * 1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys:
                slow_pts    (B, N, 3), slow_density    (B, N)
                faster_pts  (B, N, 3), faster_density  (B, N)
                fastest_pts (B, N, 3), fastest_density (B, N)

        Returns:
            logits: (B, n_classes)
        """
        f_slow    = self.scn_slow(batch["slow_pts"],    batch["slow_density"])
        f_faster  = self.scn_faster(batch["faster_pts"],  batch["faster_density"])
        f_fastest = self.scn_fastest(batch["fastest_pts"], batch["fastest_density"])

        feat = torch.cat([f_slow, f_faster, f_fastest], dim=-1)  # (B, 3072)
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# Single-scale SCN (used when slow_to_fast=False)
# ---------------------------------------------------------------------------

class SingleSCN(nn.Module):
    """Single-scale SCN with a standard classifier head."""

    def __init__(
        self,
        n_classes: int,
        n1: int = 512,
        n2: int = 128,
        k: int = 20,
        base_channels: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = SCNEncoder(n1=n1, n2=n2, k=k, base_channels=base_channels)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        feat = self.encoder(batch["points"], batch["density"])
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scn(
    n_classes: int,
    slow_to_fast: bool = True,
    n1: int = 512,
    n2: int = 128,
    k: int = 20,
    base_channels: int = 128,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build the appropriate SCN variant.

    Paper hyperparameters (Section 4.2):
        n1=512, n2=128, k=20, dropout=0.3
    """
    # Pass arguments explicitly to avoid potential typing issues with **kwargs
    if slow_to_fast:
        return SlowToFastSCN(
            n_classes=int(n_classes),
            n1=int(n1),
            n2=int(n2),
            k=int(k),
            base_channels=int(base_channels),
            dropout=float(dropout),
        )
    else:
        return SingleSCN(
            n_classes=int(n_classes),
            n1=int(n1),
            n2=int(n2),
            k=int(k),
            base_channels=int(base_channels),
            dropout=float(dropout),
        )
