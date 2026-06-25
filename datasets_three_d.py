import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from torch.utils.data import Dataset


class LungControlNetPairsCSVDataset(Dataset):
    """CSV-backed longitudinal latent pairs for ControlNet training/inference."""

    DEFAULT_FUTURE_CONTEXT_COLS = [
        "sex_male",
        "z_height_cm",
        "z_bmi",
        "z_pack_years",
        "z_log1p_lung_vol",
    ]

    def __init__(
        self,
        csv_path: str,
        patient_col: str = "patient",
        time_col: str = "timepoint",
        latent_col: str = "latent_img_path",
        mask_col: Optional[str] = "latent_mask_path",
        future_context_cols: Optional[List[str]] = None,
        baseline_age_col: str = "z_age",
        diffusion_age_col: str = "z_age",
        conditioning_mode: str = "semi",
        append_mask_token: bool = True,
        pairing: str = "all_pairs",
        pairs_per_patient_cap: Optional[int] = 10,
        dt_min: float = 0.0,
        dt_max: float = 3.0,
        identity_prob: float = 0.0,
        return_is_identity: bool = False,
        strict_paths: bool = True,
        verbose: bool = False,
        rng_seed: int = 0,
        mmap_mode: Optional[str] = None,
        clamp_min: float = -7.0,
        clamp_max: float = 7.0,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.patient_col = patient_col
        self.time_col = time_col
        self.latent_col = latent_col
        self.mask_col = mask_col
        self.future_context_cols = list(future_context_cols or self.DEFAULT_FUTURE_CONTEXT_COLS)
        self.baseline_age_col = baseline_age_col
        self.diffusion_age_col = diffusion_age_col
        self.conditioning_mode = conditioning_mode
        self.append_mask_token = bool(append_mask_token)
        self.pairing = pairing
        self.pairs_per_patient_cap = pairs_per_patient_cap
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.identity_prob = float(identity_prob)
        self.return_is_identity = bool(return_is_identity)
        self.strict_paths = bool(strict_paths)
        self.verbose = bool(verbose)
        self._rng = np.random.default_rng(int(rng_seed))
        self._mmap_mode = mmap_mode
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        if self.conditioning_mode not in {"none", "semi", "full"}:
            raise ValueError("conditioning_mode must be 'none', 'semi', or 'full'")
        if self.pairing not in {"all_pairs", "baseline_to_all"}:
            raise ValueError("pairing must be 'all_pairs' or 'baseline_to_all'")
        if not (0.0 <= self.identity_prob <= 1.0):
            raise ValueError("identity_prob must be in [0, 1]")

        required = [self.patient_col, self.time_col, self.latent_col]
        required += [self.baseline_age_col, self.diffusion_age_col]
        if self.conditioning_mode != "none":
            required += self.future_context_cols
        missing = [c for c in dict.fromkeys(required) if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        self.impute_map = {
            "sex_male": 1.0,
            self.baseline_age_col: 0.0,
            self.diffusion_age_col: 0.0,
        }
        for col in self.future_context_cols:
            self.impute_map.setdefault(col, 0.0)

        self.pairs = self._build_pairs()
        if self.verbose:
            print(f"[Dataset] Pairs: {len(self.pairs)} | Mode: {self.conditioning_mode}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_npy(self, path: str) -> np.ndarray:
        return np.load(path, allow_pickle=False, mmap_mode=self._mmap_mode)

    def _load_latent_tensor(self, path: str) -> torch.Tensor:
        latent = torch.from_numpy(np.asarray(self._load_npy(path))).float()
        latent = torch.clamp(latent, min=self.clamp_min, max=self.clamp_max)
        if latent.ndim == 5:
            latent = latent.squeeze(0)
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        if latent.ndim != 4:
            raise ValueError(f"Unexpected latent shape {tuple(latent.shape)}; expected (C,D,H,W)")
        return latent

    @staticmethod
    def _fix_mask_shape(mask: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        if mask.ndim == 5:
            mask = mask.squeeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.ndim != 4 or tuple(mask.shape[-3:]) != tuple(spatial_shape):
            return torch.ones((1,) + tuple(spatial_shape), dtype=torch.float32)
        return mask.float()

    def _build_pairs(self) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for _, group in self.df.groupby(self.patient_col, sort=False):
            if len(group) < 2:
                continue

            group = group.copy()
            group[self.time_col] = pd.to_numeric(group[self.time_col], errors="coerce")
            group = group.dropna(subset=[self.time_col]).sort_values(self.time_col)
            idxs = group.index.to_list()
            times = group[self.time_col].to_numpy(dtype=float)
            candidates: List[Tuple[int, int]] = []

            if self.pairing == "baseline_to_all":
                for k in range(1, len(idxs)):
                    dt = times[k] - times[0]
                    if dt <= self.dt_max + 0.01:
                        candidates.append((idxs[0], idxs[k]))
            else:
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        dt = times[b] - times[a]
                        if dt <= self.dt_max + 0.01:
                            candidates.append((idxs[a], idxs[b]))

            if self.pairs_per_patient_cap is not None and len(candidates) > int(self.pairs_per_patient_cap):
                sel = self._rng.choice(len(candidates), size=int(self.pairs_per_patient_cap), replace=False)
                candidates = [candidates[int(i)] for i in sel]

            pairs.extend(candidates)
        return pairs

    def _normalize_dt(self, dt: float) -> float:
        denom = max(1e-8, self.dt_max - self.dt_min)
        return float(np.clip((float(dt) - self.dt_min) / denom, 0.0, 1.0))

    def _get_val_imputed(self, row: pd.Series, col: str) -> float:
        val = row.get(col, np.nan)
        if pd.isna(val):
            return float(self.impute_map.get(col, 0.0))
        return float(val)

    def _build_controlnet_context(self, base_row, followup_row, dt_norm: float) -> torch.Tensor:
        if self.conditioning_mode == "none":
            return torch.zeros((1, self.controlnet_context_dim), dtype=torch.float32)

        vals = [self._get_val_imputed(followup_row, c) for c in self.future_context_cols]
        vals.append(self._get_val_imputed(base_row, self.baseline_age_col))
        vals.append(float(dt_norm))
        if self.append_mask_token:
            vals.append(1.0)
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

    def _build_diffusion_context(self, followup_row) -> torch.Tensor:
        if self.conditioning_mode == "none":
            return torch.zeros((1, self.diffusion_context_dim), dtype=torch.float32)

        vals = [self._get_val_imputed(followup_row, self.diffusion_age_col)]
        vals += [self._get_val_imputed(followup_row, c) for c in self.future_context_cols]
        if self.append_mask_token:
            vals.append(1.0)
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i, j = self.pairs[idx]
        base_row = self.df.iloc[i]
        followup_row = self.df.iloc[j]

        is_identity = bool(torch.rand(()) < self.identity_prob)
        if is_identity:
            followup_row = base_row
            dt_norm = 0.0
        else:
            dt = max(0.0, float(followup_row[self.time_col]) - float(base_row[self.time_col]))
            dt_norm = self._normalize_dt(dt)

        start_path = base_row[self.latent_col]
        followup_path = start_path if is_identity else followup_row[self.latent_col]
        if self.strict_paths:
            if not isinstance(start_path, str) or not os.path.exists(start_path):
                raise FileNotFoundError(f"Missing starting latent file: {start_path}")
            if not isinstance(followup_path, str) or not os.path.exists(followup_path):
                raise FileNotFoundError(f"Missing follow-up latent file: {followup_path}")

        start_latent = self._load_latent_tensor(start_path)
        followup_latent = start_latent.clone() if is_identity else self._load_latent_tensor(followup_path)

        sample = {
            "starting_latent": start_latent,
            "followup_latent": followup_latent,
            "context": self._build_controlnet_context(base_row, followup_row, dt_norm),
            "diffusion_context": self._build_diffusion_context(followup_row),
            "starting_time": torch.tensor([dt_norm], dtype=torch.float32),
            "delta_time": torch.tensor([dt_norm], dtype=torch.float32),
        }

        if self.return_is_identity:
            sample["is_identity"] = torch.tensor([1.0 if is_identity else 0.0], dtype=torch.float32)

        if self.mask_col is not None and self.mask_col in self.df.columns:
            mask_path = followup_row.get(self.mask_col, None)
            if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
                mask = torch.from_numpy(np.asarray(self._load_npy(mask_path))).float()
                sample["mask"] = self._fix_mask_shape(mask, followup_latent.shape[-3:])
            else:
                sample["mask"] = torch.ones((1,) + tuple(followup_latent.shape[-3:]), dtype=torch.float32)

        return sample

    @property
    def controlnet_context_dim(self) -> int:
        return len(self.future_context_cols) + 2 + (1 if self.append_mask_token else 0)

    @property
    def diffusion_context_dim(self) -> int:
        return 1 + len(self.future_context_cols) + (1 if self.append_mask_token else 0)


class LungLatentCSVDataset(Dataset):
    """CSV-backed single-scan latent dataset for latent diffusion pretraining."""

    DEFAULT_CONTEXT_COLS = [
        "z_age",
        "sex_male",
        "z_height_cm",
        "z_bmi",
        "z_pack_years",
        "z_log1p_lung_vol",
    ]

    def __init__(
        self,
        csv_path: str,
        context_cols=None,
        conditioning_mode: str = "semi",
        latent_col: str = "latent_img_path",
        mask_col: str = "latent_mask_path",
        append_mask_token: bool = True,
        strict_paths: bool = True,
        verbose: bool = False,
        clamp_min: float = -7.0,
        clamp_max: float = 7.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.conditioning_mode = conditioning_mode
        self.latent_col = latent_col
        self.mask_col = mask_col
        self.append_mask_token = bool(append_mask_token)
        self.strict_paths = bool(strict_paths)
        self.verbose = bool(verbose)
        self.context_cols = list(context_cols or self.DEFAULT_CONTEXT_COLS)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        if self.conditioning_mode not in {"none", "full", "semi"}:
            raise ValueError("conditioning_mode must be one of 'none', 'full', or 'semi'")
        if self.latent_col not in self.df.columns:
            raise ValueError(f"CSV missing required latent column '{self.latent_col}'")
        if self.mask_col is not None and self.mask_col not in self.df.columns:
            raise ValueError(f"CSV missing mask column '{self.mask_col}' (or set mask_col=None)")
        if self.conditioning_mode != "none":
            missing = [c for c in self.context_cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"CSV missing context columns: {missing}")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _fix_latent_shape(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 5:
            latent = latent.squeeze(0)
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        if latent.ndim != 4:
            raise ValueError(f"Unexpected latent shape {tuple(latent.shape)}; expected (C,D,H,W)")
        return latent

    @staticmethod
    def _fix_mask_shape(mask: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        if mask.ndim == 5:
            mask = mask.squeeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.ndim != 4 or tuple(mask.shape[-3:]) != tuple(spatial_shape):
            return torch.ones((1,) + tuple(spatial_shape), dtype=torch.float32)
        return mask.float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        latent_path = row[self.latent_col]
        if self.strict_paths and (not isinstance(latent_path, str) or not os.path.exists(latent_path)):
            raise FileNotFoundError(f"Missing latent file at row {idx}: {latent_path}")

        latent = torch.from_numpy(np.asarray(np.load(latent_path, allow_pickle=False))).float()
        latent = torch.clamp(latent, min=self.clamp_min, max=self.clamp_max)
        latent = self._fix_latent_shape(latent)
        sample = {"latent": latent}

        if self.mask_col is not None:
            mask_path = row[self.mask_col]
            if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
                mask = torch.from_numpy(np.asarray(np.load(mask_path, allow_pickle=False))).float()
                sample["mask"] = self._fix_mask_shape(mask, latent.shape[-3:])
            else:
                sample["mask"] = torch.ones((1,) + tuple(latent.shape[-3:]), dtype=torch.float32)
                if self.verbose:
                    print(f"[mask] missing at row {idx}: {mask_path} -> using ones")

        if self.conditioning_mode == "none":
            ctx_dim = len(self.context_cols) + (1 if self.append_mask_token else 0)
            sample["context"] = torch.zeros((1, ctx_dim), dtype=torch.float32)
            return sample

        context_values = []
        any_missing = False
        for col in self.context_cols:
            val = row[col]
            if pd.isna(val):
                any_missing = True
                context_values.append(0.0)
            else:
                context_values.append(float(val))

        if self.conditioning_mode == "full" and any_missing:
            raise ValueError(f"Found NaN covariates in conditioning_mode='full' at row {idx}")

        if self.append_mask_token:
            mask_token = 1.0 if self.conditioning_mode == "full" or not any_missing else 0.0
            if self.conditioning_mode == "semi" and mask_token == 0.0:
                context_values = [0.0] * len(self.context_cols)
            context_values.append(mask_token)

        sample["context"] = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)
        return sample


class Custom3DDDatasetMask(Dataset):
    """Preprocessed CT .npz dataset with optional aligned lung masks for AE patches."""

    def __init__(
        self,
        txt_path,
        transform=None,
        use_lung_mask: bool = False,
        summit_only: bool = False,
        summit_substr: str = "summit",
        verbose_mask_warnings: bool = False,
        assume_npz_has_affine: bool = True,
    ):
        with open(txt_path, "r") as f:
            samples = [line.strip() for line in f if line.strip()]

        if summit_only:
            key = summit_substr.lower()
            samples = [p for p in samples if key in p.lower()]
            print(f"[dataset] summit_only=True -> kept {len(samples)} samples containing '{summit_substr}'")

        if not samples:
            raise RuntimeError(f"[dataset] No samples found after loading '{txt_path}'")

        self.samples = samples
        self.transform = transform
        self.use_lung_mask = use_lung_mask
        self.verbose_mask_warnings = verbose_mask_warnings
        self.assume_npz_has_affine = assume_npz_has_affine
        self._warned_transform_bad_return = False

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _build_mask_path(img_path: str) -> str:
        return img_path.replace("/img/", "/mask/").replace("_resampled.npz", "_lungmask.nii.gz")

    def _load_lung_mask_aligned(self, mask_path: str, vol_shape, vol_affine):
        if not os.path.exists(mask_path):
            if self.verbose_mask_warnings:
                print(f"[lung-mask] Missing: {mask_path} -> using full mask.")
            return np.ones(vol_shape, dtype=np.float32)

        try:
            nii = nib.load(mask_path)
            mask_data = np.asarray(nii.get_fdata(dtype=np.float32))
        except Exception as exc:
            if self.verbose_mask_warnings:
                print(f"[lung-mask] Error loading {mask_path}: {exc} -> using full mask.")
            return np.ones(vol_shape, dtype=np.float32)

        if mask_data.ndim == 4:
            mask_data = np.squeeze(mask_data)

        try:
            xform = ornt_transform(io_orientation(nii.affine), io_orientation(vol_affine))
            mask_data = apply_orientation(mask_data, xform)
        except Exception as exc:
            if self.verbose_mask_warnings:
                print(f"[lung-mask] Orientation align failed for {mask_path}: {exc} (using raw array).")

        if mask_data.shape != vol_shape:
            if self.verbose_mask_warnings:
                print(f"[lung-mask] Shape mismatch: mask {mask_data.shape} vs vol {vol_shape} -> full mask.")
            return np.ones(vol_shape, dtype=np.float32)

        return (mask_data > 0.5).astype(np.float32)

    def __getitem__(self, idx):
        path = self.samples[idx]
        with np.load(path) as arr:
            vol01 = arr["data"].astype(np.float32)
            if self.assume_npz_has_affine and "affine" in arr:
                vol_affine = np.asarray(arr["affine"], dtype=np.float32)
            else:
                vol_affine = np.eye(4, dtype=np.float32)

        vol = np.clip(vol01 * 2.0 - 1.0, -1.0, 1.0)
        x = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0).float()

        lung_mask = None
        if self.use_lung_mask:
            mask_path = self._build_mask_path(path)
            lung_mask_np = self._load_lung_mask_aligned(mask_path, vol.shape, vol_affine)
            lung_mask = torch.from_numpy(np.ascontiguousarray(lung_mask_np)).unsqueeze(0).float()
            lung_mask = (lung_mask > 0.5).float()

        if self.transform is not None:
            if self.use_lung_mask:
                out = self.transform((x, lung_mask))
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    x, lung_mask = out
                else:
                    if not self._warned_transform_bad_return:
                        print("[WARN] transform((x, lung_mask)) did not return (x, lung_mask). Keeping mask.")
                        self._warned_transform_bad_return = True
                    x = out
            else:
                x = self.transform(x)

        return (x, lung_mask) if self.use_lung_mask else x
