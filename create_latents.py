#!/usr/bin/env python3
"""
Extract stitched latent MU volumes (and downsampled lung masks) from FULL .nii.gz CT volumes.

- Input volumes: .nii.gz (HU, NOT min-max normalized)
- Normalization: clean non-finite -> HU clip [-1200, 800] -> [0,1] -> [-1,1]
- Mask: loads a .nii.gz lung mask, reorients to match volume if needed, keeps binary, then downsamples to latent grid.
- Encoding: patch-wise on GPU, stitching on CPU to reduce OOM risk.
- Output:
    <output_dir>/latents/<stem>.npy   (float16) shape [C, D/4, H/4, W/4]
    <output_dir>/masks/<stem>.npy     (float16) shape [1, D/4, H/4, W/4]
  plus optional: text lists of final saved latent/mask paths (can be on non-scratch storage).

Orientation handling:
  - The CT volume is canonicalized with nib.as_closest_canonical() BEFORE extracting the voxel array.
  - The mask is also canonicalized and then reoriented (as a safety net) to match the canonical CT orientation.
  - This keeps latents in a consistent orientation across cohorts.

Registered preprocessing naming convention:
  IMG : .../registered/img/.../<name>_ct_256.nii.gz
  MASK: .../registered/mask/.../<name>_mask_256.nii.gz

Use:
  --mask_mode registered
  --require_mask   (optional, but recommended to avoid silent full-mask fallback)
"""
import os
import argparse
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

from hybrid_vae_vitconv import ConvAEAdapter


# ============================================================
# Dataset: NIfTI volume (HU) + aligned lung mask (NIfTI)
# ============================================================
class NiftiDatasetWithMask(Dataset):
    """
    mask_mode:
      - "registered": assumes paired convention:
            /.../registered/img/.../<stem>_ct_256.nii.gz
            /.../registered/mask/.../<stem>_mask_256.nii.gz
        (derives mask by replacing "/img/"->"/mask/" and "_ct_256.nii.gz"->"_mask_256.nii.gz")

      - "derive": generic derive mode using --mask_dir_replace and --mask_suffix
      - "explicit": mask paths are provided in a separate txt (same length as images)
      - "none": use full mask of ones
    """
    def __init__(
        self,
        txt_path: str,
        mask_mode: str = "registered",       # {"registered","derive","none","explicit"}
        mask_suffix: str = "_lungmask.nii.gz",
        mask_dir_replace: str = "/img/->/mask/",
        explicit_mask_txt: Optional[str] = None,
        verbose: bool = False,
        hu_min: float = -1200.0,
        hu_max: float = 800.0,
        img_already_01: bool = False,
        canonicalize: bool = True,
    ):
        with open(txt_path, "r") as f:
            self.files = [ln.strip() for ln in f if ln.strip()]

        self.verbose = verbose
        self.mask_mode = mask_mode
        self.mask_suffix = mask_suffix
        self.mask_dir_replace = mask_dir_replace
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.img_already_01 = bool(img_already_01)
        self.canonicalize = bool(canonicalize)

        self.explicit_masks: Optional[List[str]] = None
        if mask_mode == "explicit":
            if explicit_mask_txt is None:
                raise ValueError("mask_mode='explicit' requires --explicit_mask_txt")
            with open(explicit_mask_txt, "r") as f:
                masks = [ln.strip() for ln in f if ln.strip()]
            if len(masks) != len(self.files):
                raise ValueError(f"explicit mask list length mismatch: {len(masks)} masks vs {len(self.files)} images")
            self.explicit_masks = masks

        if mask_mode not in {"registered", "derive", "none", "explicit"}:
            raise ValueError(f"Unknown mask_mode={mask_mode}")

    def _derive_mask_path_generic(self, img_path: str) -> str:
        s = str(img_path)

        # replace directory fragment (e.g. /img/ -> /mask/)
        if "->" in self.mask_dir_replace:
            a, b = self.mask_dir_replace.split("->", 1)
            s = s.replace(a, b)

        # remove extension
        if s.endswith(".nii.gz"):
            base = s[:-7]
        elif s.endswith(".nii"):
            base = s[:-4]
        else:
            base = os.path.splitext(s)[0]

        # Strip common preprocessed-image suffixes before adding the mask suffix.
        if base.endswith("_resampled"):
            base = base[:-len("_resampled")]

        # now append mask suffix (e.g. "_lungmask.nii.gz")
        return base + self.mask_suffix



    def _derive_mask_path_registered(self, img_path: str) -> str:
        s = str(img_path).replace("/img/", "/mask/")
        if s.endswith("_ct_256.nii.gz"):
            return s[:-len("_ct_256.nii.gz")] + "_mask_256.nii.gz"

        if s.endswith("_ct.nii.gz"):
            return s[:-len("_ct.nii.gz")] + "_mask.nii.gz"
        if s.endswith(".nii.gz"):
            return self._derive_mask_path_generic(img_path)

        return self._derive_mask_path_generic(img_path)

    def _load_mask_aligned(self, mask_path: str, vol_shape, vol_affine):
        """
        Load mask, canonicalize (optional), then reorient to match vol_affine orientation.
        Returns a binary mask matching vol_shape, else returns full mask.
        """
        if not os.path.exists(mask_path):
            if self.verbose:
                print(f"[mask] missing {mask_path} -> full mask")
            return np.ones(vol_shape, np.float32)

        mnii = nib.load(mask_path)

        if self.canonicalize:
            try:
                mnii = nib.as_closest_canonical(mnii)
            except Exception as e:
                if self.verbose:
                    print(f"[mask] canonicalize failed {mask_path}: {e}")

        mask = np.asarray(mnii.get_fdata(dtype=np.float32))
        if mask.ndim == 4:
            mask = np.squeeze(mask)

        # Reorient mask array to match volume orientation (safe even if already aligned)
        try:
            ornt_mask = io_orientation(mnii.affine)
            ornt_vol  = io_orientation(vol_affine)
            xform = ornt_transform(ornt_mask, ornt_vol)
            mask = apply_orientation(mask, xform)
        except Exception as e:
            if self.verbose:
                print(f"[mask] orient failed {mask_path}: {e}")

        if mask.shape != vol_shape:
            if self.verbose:
                print(f"[mask] shape mismatch {mask.shape} vs {vol_shape} -> full mask")
            return np.ones(vol_shape, np.float32)

        return (mask > 0.5).astype(np.float32)

    def _get_mask_path(self, idx: int, img_path: str) -> str:
        if self.mask_mode == "none":
            return ""
        if self.mask_mode == "explicit":
            assert self.explicit_masks is not None
            return self.explicit_masks[idx]
        if self.mask_mode == "registered":
            return self._derive_mask_path_registered(img_path)
        return self._derive_mask_path_generic(img_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        img_path = self.files[idx]

        nii = nib.load(img_path)

        # Canonicalize CT orientation before extracting the voxel array.
        if self.canonicalize:
            try:
                nii = nib.as_closest_canonical(nii)
            except Exception as e:
                if self.verbose:
                    print(f"[img] canonicalize failed {img_path}: {e}")

        vol = np.asarray(nii.get_fdata(dtype=np.float32))
        if vol.ndim == 4:
            vol = np.squeeze(vol)

        vol_affine = np.asarray(nii.affine, dtype=np.float32)

        # Normalize HU -> [0,1] -> [-1,1]
        if self.img_already_01:
            vol = np.nan_to_num(vol, nan=0.0, posinf=1.0, neginf=0.0)
            vol01 = np.clip(vol, 0.0, 1.0)
        else:
            vol = np.nan_to_num(vol, nan=self.hu_min, posinf=self.hu_max, neginf=self.hu_min)
            np.clip(vol, self.hu_min, self.hu_max, out=vol)
            vol01 = (vol - self.hu_min) / (self.hu_max - self.hu_min)
            np.clip(vol01, 0.0, 1.0, out=vol01)

        vol_pm1 = vol01 * 2.0 - 1.0

        # x: [1, D, H, W] CPU
        x = torch.from_numpy(vol_pm1).unsqueeze(0)

        # Mask
        mask_path = self._get_mask_path(idx, img_path)
        if self.mask_mode == "none":
            mask_np = np.ones_like(vol_pm1, dtype=np.float32)
        else:
            mask_np = self._load_mask_aligned(mask_path, vol_pm1.shape, vol_affine)

        m = torch.from_numpy(mask_np).unsqueeze(0)  # [1, D, H, W]
        return x, m, img_path, mask_path


# ============================================================
# Stitching helpers (CPU stitching, patch->GPU)
# ============================================================
def _compute_positions(full: int, roi: int, stride: int):
    if roi > full:
        return [0]
    pos = list(range(0, full - roi + 1, stride))
    if pos[-1] != full - roi:
        pos.append(full - roi)
    return pos


@torch.no_grad()
def encode_volume_to_latent_mu_cpu_stitch(
    model,
    x_pm1_cpu,                       # [1,1,D,H,W] on CPU
    roi_size=96,
    overlap=0.25,
    downsample=4,
    use_amp=True,
    device=torch.device("cuda"),
    denom_eps: float = 1e-3,
):
    if isinstance(device, str):
        device = torch.device(device)

    B, C, D, H, W = x_pm1_cpu.shape
    assert B == 1 and C == 1, f"Expected [1,1,D,H,W], got {x_pm1_cpu.shape}"

    stride = int(round(roi_size * (1.0 - overlap)))
    stride = max(downsample, (stride // downsample) * downsample)

    z_starts = _compute_positions(D, roi_size, stride)
    y_starts = _compute_positions(H, roi_size, stride)
    x_starts = _compute_positions(W, roi_size, stride)

    amp_ok = (use_amp and device.type == "cuda")

    # Probe one patch to get latent channels
    patch0 = x_pm1_cpu[..., :roi_size, :roi_size, :roi_size].to(device, non_blocking=True)
    if amp_ok:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out0 = model(patch0, mask_ratio=0.0, use_mean=True)
    else:
        out0 = model(patch0, mask_ratio=0.0, use_mean=True)

    mu0 = out0[2]
    C_lat = int(mu0.shape[1])

    ld, lh, lw = D // downsample, H // downsample, W // downsample
    lroi = roi_size // downsample

    latent_cpu = torch.zeros((1, C_lat, ld, lh, lw), device="cpu", dtype=torch.float32)
    weight_cpu = torch.zeros_like(latent_cpu)

    w_patch = torch.ones((1, 1, lroi, lroi, lroi), device="cpu", dtype=torch.float32)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch_cpu = x_pm1_cpu[..., z0:z0+roi_size, y0:y0+roi_size, x0:x0+roi_size]
                patch_gpu = patch_cpu.to(device, non_blocking=True)

                if amp_ok:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(patch_gpu, mask_ratio=0.0, use_mean=True)
                else:
                    out = model(patch_gpu, mask_ratio=0.0, use_mean=True)

                mu = out[2].float().cpu()

                lz0, ly0, lx0 = z0 // downsample, y0 // downsample, x0 // downsample
                latent_cpu[..., lz0:lz0+lroi, ly0:ly0+lroi, lx0:lx0+lroi] += mu * w_patch
                weight_cpu[..., lz0:lz0+lroi, ly0:ly0+lroi, lx0:lx0+lroi] += w_patch

                del patch_gpu, out

    latent = latent_cpu / weight_cpu.clamp_min(float(denom_eps))
    latent = torch.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0)
    return latent


def downsample_mask_to_latent(mask_cpu, downsample_factor=4):
    return F.interpolate(mask_cpu, scale_factor=1.0/downsample_factor, mode="nearest")


# ============================================================
# Checkpoint loader
# ============================================================
def load_full_model_checkpoint(model, path):
    print(f"[load] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"[load] missing: {len(msg.missing_keys)} | unexpected: {len(msg.unexpected_keys)}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Encode 256^3 CT volumes into stitched 4 x 64^3 latents.")

    ap.add_argument("--data_list", type=str, required=True,
                    help="TXT with one registered CT .nii/.nii.gz path per line")

    ap.add_argument("--output_dir", type=str,
                    default="outputs/latents",
                    help="Scratch output dir (latents/ and masks/ subfolders will be created)")

    ap.add_argument("--final_output_dir", type=str,
                    default=None,
                    help="If set, copy latents/masks here after creation (persistent storage)")

    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Autoencoder checkpoint from main_pretrain.py")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--roi", type=int, default=96)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--verbose_masks", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="If set, recompute even if files exist.")

    # Mask options
    ap.add_argument("--mask_mode", type=str, default="derive",
                    choices=["registered", "derive", "none", "explicit"])
    ap.add_argument("--mask_suffix", type=str, default="_lungmask.nii.gz")
    ap.add_argument("--mask_dir_replace", type=str, default="/img/->/mask/")
    ap.add_argument("--explicit_mask_txt", type=str, default=None)
    ap.add_argument("--require_mask", action="store_true",
                    help="If set, skip cases where derived/registered/explicit mask file is missing.")

    # Image normalization options
    ap.add_argument("--img_already_01", action="store_true",
                    help="If set, assumes image is already in [0,1] (skip HU normalization).")
    ap.add_argument("--hu_min", type=float, default=-1200.0)
    ap.add_argument("--hu_max", type=float, default=800.0)

    ap.add_argument("--no_canonicalize", action="store_true",
                    help="If set, do NOT apply nib.as_closest_canonical() before encoding (not recommended).")

    # Lists (persistent)
    ap.add_argument("--latent_list_out", type=str,
                    default=None,
                    help="Write final latent .npy paths (one per line)")
    ap.add_argument("--mask_list_out", type=str,
                    default=None,
                    help="Write final mask .npy paths (one per line)")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    latent_dir = os.path.join(args.output_dir, "latents")
    mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    do_copy = bool(args.final_output_dir)
    if do_copy:
        os.makedirs(args.final_output_dir, exist_ok=True)
        final_latent_dir = os.path.join(args.final_output_dir, "latents")
        final_mask_dir = os.path.join(args.final_output_dir, "masks")
        os.makedirs(final_latent_dir, exist_ok=True)
        os.makedirs(final_mask_dir, exist_ok=True)
    else:
        final_latent_dir = latent_dir
        final_mask_dir = mask_dir

    model = ConvAEAdapter(
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        num_res_blocks=(2, 2, 2),
        latent_channels=4,
        attention_levels=(False, False, False),
        norm_num_groups=32,
        use_checkpointing=False,
        use_convtranspose=False,
        output_sigmoid=False,
    ).to(device).eval()

    load_full_model_checkpoint(model, args.checkpoint)

    dataset = NiftiDatasetWithMask(
        txt_path=args.data_list,
        mask_mode=args.mask_mode,
        mask_suffix=args.mask_suffix,
        mask_dir_replace=args.mask_dir_replace,
        explicit_mask_txt=args.explicit_mask_txt,
        verbose=args.verbose_masks,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        img_already_01=args.img_already_01,
        canonicalize=(not args.no_canonicalize),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    running_sum = torch.zeros((), dtype=torch.float64)
    running_sq = torch.zeros((), dtype=torch.float64)
    total_vox = 0

    latent_written: List[str] = []
    mask_written: List[str] = []

    print(f"[Run] Processing {len(dataset)} volumes...")
    print(f"[Out] Scratch: {args.output_dir}")
    if do_copy:
        print(f"[Out] Final:   {args.final_output_dir}")
    print(f"[Ornt] Canonicalize before encoding: {not args.no_canonicalize}")

    for x, mask, img_path_list, mask_path_list in tqdm(loader):
        img_path = img_path_list[0]
        mask_path = mask_path_list[0]

        if args.require_mask and args.mask_mode != "none":
            if not mask_path or (not os.path.exists(mask_path)):
                print(f"[skip] missing mask for {img_path} -> {mask_path}")
                continue

        patient_id = Path(img_path).parent.name
        bname = os.path.basename(img_path)
        stem = bname[:-7] if bname.endswith(".nii.gz") else os.path.splitext(bname)[0]
        out_name = f"{patient_id}_{stem}.npy"

        scratch_lat = os.path.join(latent_dir, out_name)
        scratch_msk = os.path.join(mask_dir, out_name)

        final_lat = os.path.join(final_latent_dir, out_name)
        final_msk = os.path.join(final_mask_dir, out_name)

        if (not args.overwrite) and os.path.exists(final_lat) and os.path.exists(final_msk):
            latent_written.append(final_lat)
            mask_written.append(final_msk)
            continue

        x = x.contiguous()
        mask = mask.contiguous()

        try:
            latent = encode_volume_to_latent_mu_cpu_stitch(
                model=model,
                x_pm1_cpu=x,
                roi_size=args.roi,
                overlap=args.overlap,
                downsample=args.downsample,
                use_amp=args.use_amp,
                device=device,
                denom_eps=1e-3,
            )
        except RuntimeError as e:
            print(f"[Error] Failed on {out_name}: {e}")
            continue

        latent_mask = downsample_mask_to_latent(mask.float(), downsample_factor=args.downsample)

        np.save(scratch_lat, latent.squeeze(0).numpy().astype(np.float16))
        np.save(scratch_msk, latent_mask.squeeze(0).numpy().astype(np.float16))

        if do_copy:
            shutil.copy2(scratch_lat, final_lat)
            shutil.copy2(scratch_msk, final_msk)

        latent_written.append(final_lat)
        mask_written.append(final_msk)

        lf = latent.float()
        running_sum += lf.sum().double()
        running_sq += (lf * lf).sum().double()
        total_vox += lf.numel()

    if total_vox > 0:
        mean = running_sum / total_vox
        var = (running_sq / total_vox) - (mean * mean)
        std = torch.sqrt(var.clamp_min(1e-12))
        print("\n" + "=" * 70)
        print(f"Latent mean: {mean.item():.6f}")
        print(f"Latent std:  {std.item():.6f}")
        print(f"Scale factor (1/std): {(1.0 / std.item()):.6f}")
        print("=" * 70)

    if args.latent_list_out:
        Path(os.path.dirname(args.latent_list_out)).mkdir(parents=True, exist_ok=True)
        with open(args.latent_list_out, "w") as f:
            for p in latent_written:
                f.write(p + "\n")
        print(f"[OK] Wrote latent list: {args.latent_list_out} ({len(latent_written)} entries)")

    if args.mask_list_out:
        Path(os.path.dirname(args.mask_list_out)).mkdir(parents=True, exist_ok=True)
        with open(args.mask_list_out, "w") as f:
            for p in mask_written:
                f.write(p + "\n")
        print(f"[OK] Wrote mask list: {args.mask_list_out} ({len(mask_written)} entries)")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
