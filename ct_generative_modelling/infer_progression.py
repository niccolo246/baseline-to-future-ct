#!/usr/bin/env python3
"""Generate proposed-model longitudinal CT predictions as NIfTI volumes."""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generative.networks.schedulers import DDIMScheduler
from ct_generative_modelling.networks_shallow import init_controlnet_ct, init_latent_diffusion_ct
from hybrid_vae_vitconv import ConvAEAdapter


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _sanitize_for_path(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s[:128] if len(s) > 128 else s


def _get_affine_and_header(
    ref_nii: str,
    default_spacing: tuple = (1.35, 1.35, 1.35)
):
    """
    Returns (affine, header) consistent with how voxel arrays are typically handled.
    - If ref_nii exists:
        * load it
        * convert to closest canonical orientation
        * return its affine + header
    - Else: fallback to identity with default spacing.
    """
    if ref_nii is not None and os.path.exists(ref_nii):
        r = nib.load(ref_nii)
        r = nib.as_closest_canonical(r)

        return np.asarray(r.affine, dtype=np.float32), r.header.copy()

    sx, sy, sz = default_spacing
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = sx
    affine[1, 1] = sy
    affine[2, 2] = sz
    return affine, None



def resolve_ref_img_path(mask_path_str: str) -> str:
    """
    Converts /IPF/mask/123_lungmask.nii.gz -> /IPF/img/123_resampled.nii.gz
    """
    if not isinstance(mask_path_str, str):
        return None
    p = mask_path_str
    p = p.replace("IPF/mask", "IPF/img")
    p = p.replace("_lungmask", "_resampled")
    return p


def unit_to_hu(vol_unit_01: np.ndarray) -> np.ndarray:
    vol_unit_01 = np.clip(vol_unit_01, 0.0, 1.0)
    hu = vol_unit_01 * 2000.0 - 1200.0
    return np.clip(hu, -1200.0, 800.0)


def decoded_minus1_1_to_hu(vol_minus1_1: torch.Tensor) -> np.ndarray:
    v = torch.clamp(vol_minus1_1, -1.0, 1.0)
    v01 = (v + 1.0) / 2.0
    v01_np = v01.detach().cpu().numpy().astype(np.float32)
    hu = unit_to_hu(v01_np)
    return hu.astype(np.float32)


def save_nii_gz(
    hu_vol: np.ndarray,
    affine: np.ndarray,
    header,
    out_path: str,
    save_int16: bool = True
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if save_int16:
        hu_to_save = np.rint(np.clip(hu_vol, -1200.0, 800.0)).astype(np.int16)
    else:
        hu_to_save = np.clip(hu_vol, -1200.0, 800.0).astype(np.float32)

    hdr = header.copy() if header is not None else None

    # Match reconstruct_from_latent.py behavior:
    # create image using reference header and do NOT forcibly overwrite sform/qform.
    img = nib.Nifti1Image(hu_to_save, affine.astype(np.float32), hdr)

    # Only set xforms if we *don't* have a reference header (fallback case).
    if hdr is None:
        img.set_sform(affine.astype(np.float32), code=1)
        img.set_qform(affine.astype(np.float32), code=1)

    nib.save(img, out_path)



def _strip_module_prefix(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _select_state_dict(ck: object, prefer_ema: bool = False) -> dict:
    """
    Handles common checkpoint layouts:
      - {"model_without_ddp": state_dict}
      - {"model": state_dict}
      - {"state_dict": state_dict}
      - raw state_dict
      - ControlNet EMA: {"ema_state": {"shadow": state_dict}} or {"ema_state": state_dict}
    """
    if isinstance(ck, dict):
        # EMA (ControlNet)
        if prefer_ema and "ema_state" in ck:
            ema = ck["ema_state"]
            if isinstance(ema, dict) and "shadow" in ema and isinstance(ema["shadow"], dict):
                return ema["shadow"]
            if isinstance(ema, dict):
                # sometimes ema_state is itself the state dict
                # or nested in some other form
                # if it looks like state_dict, return it
                if any(isinstance(v, torch.Tensor) for v in ema.values()):
                    return ema
            # fallback: try raw
        # Standard layouts
        if "model_without_ddp" in ck and isinstance(ck["model_without_ddp"], dict):
            return ck["model_without_ddp"]
        if "model" in ck and isinstance(ck["model"], dict):
            return ck["model"]
        if "state_dict" in ck and isinstance(ck["state_dict"], dict):
            return ck["state_dict"]

        # Sometimes checkpoints store a nested dict under another key;
        # if the dict itself looks like a state_dict (tensor values), accept it
        if any(isinstance(v, torch.Tensor) for v in ck.values()):
            return ck

    # If it's not a dict or not a recognized layout, assume it's already a state dict
    return ck


def load_weights_flex(model, ckpt_path: str, *, prefer_ema: bool = False, name: str = "model"):
    if ckpt_path is None or ckpt_path == "":
        print(f"[Load] {name}: no checkpoint provided.")
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[Load] {name}: checkpoint not found: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _select_state_dict(ck, prefer_ema=prefer_ema)
    if not isinstance(sd, dict):
        raise ValueError(f"[Load] {name}: selected state_dict is not a dict (got {type(sd)}).")

    sd = _strip_module_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f"[Load] {name}: {os.path.basename(ckpt_path)}")
    print(f"       missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print(f"       first missing: {missing[:12]}")
    if len(unexpected) > 0:
        print(f"       first unexpected: {unexpected[:12]}")

    if len(missing) > 500:
        print(f"[Load] WARNING: {name} has a very large number of missing keys "
              f"({len(missing)}). This often means weights were not compatible "
              f"or the wrong checkpoint layout was used.")


@torch.no_grad()
def _decode_latent_patchwise_robust(ae, z, patch_size=24, overlap=0.25):
    device = z.device
    z = z.contiguous()
    _, _, D, H, W = z.shape
    OD, OH, OW = D * 4, H * 4, W * 4

    out = torch.zeros((1, 1, OD, OH, OW), device="cpu", dtype=torch.float32)
    wgt = torch.zeros_like(out)

    stride = int(round(patch_size * (1.0 - overlap)))
    stride = max(1, stride)

    def get_starts(dim, psize, stride_):
        starts = list(range(0, dim - psize + 1, stride_))
        if starts[-1] != dim - psize:
            starts.append(dim - psize)
        return starts

    zs = get_starts(D, patch_size, stride)
    ys = get_starts(H, patch_size, stride)
    xs = get_starts(W, patch_size, stride)

    model_dtype = next(ae.parameters()).dtype

    # Init weight patch
    zp_probe = z[..., :patch_size, :patch_size, :patch_size].to(device, dtype=model_dtype)
    with torch.cuda.amp.autocast(enabled=False):
        yp_probe = ae.ae.decode(zp_probe) if hasattr(ae, "ae") else ae.decode(zp_probe)
        if isinstance(yp_probe, (tuple, list)):
            yp_probe = yp_probe[0]
    pD, pH, pW = yp_probe.shape[2:]
    w_patch = torch.ones((1, 1, pD, pH, pW), device="cpu")

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                zp = z[..., z0:z0 + patch_size, y0:y0 + patch_size, x0:x0 + patch_size].to(
                    device, dtype=model_dtype
                ).contiguous()

                with torch.cuda.amp.autocast(enabled=False):
                    yp = ae.ae.decode(zp) if hasattr(ae, "ae") else ae.decode(zp)
                    if isinstance(yp, (tuple, list)):
                        yp = yp[0]

                yp = yp.cpu().float()
                zp0, yp0_, xp0 = z0 * 4, y0 * 4, x0 * 4
                curr_pD, curr_pH, curr_pW = yp.shape[2:]

                out[..., zp0:zp0 + curr_pD, yp0_:yp0_ + curr_pH, xp0:xp0 + curr_pW] += yp
                wgt[..., zp0:zp0 + curr_pD, yp0_:yp0_ + curr_pH, xp0:xp0 + curr_pW] += w_patch[..., :curr_pD, :curr_pH, :curr_pW]

    return out / wgt.clamp_min(1e-3)


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    print("[Vis] NIfTI export mode with EXACT REFERENCE GEOMETRY.")
    print(f"[Vis] Control Scale: {args.control_scale} | Guidance: {args.guidance_scale}")
    print(f"[Vis] Steps: {args.num_inference_steps}")
    print(f"[Vis] Latent averaging samples: {args.latent_average_samples}")

    # --- Load Models ---
    print("[Model] Loading VAE...")
    ae = ConvAEAdapter(
        in_channels=1, out_channels=1, num_channels=(64, 128, 256),
        num_res_blocks=(2, 2, 2), latent_channels=4, attention_levels=(False, False, False),
        norm_num_groups=32, use_checkpointing=True, use_convtranspose=False, output_sigmoid=False
    ).to(device)
    if args.ae_ckpt:
        load_weights_flex(ae, args.ae_ckpt, prefer_ema=False, name="VAE/AE")
    ae.eval()

    print("[Model] Loading Diffusion...")
    args.conditioning_mode = "full"
    diffusion = init_latent_diffusion_ct(
        checkpoints_path=None, args=args, context_dim=int(args.diffusion_context_dim)
    ).to(device)
    load_weights_flex(diffusion, args.diff_ckpt, prefer_ema=False, name="Diffusion")
    diffusion.eval()

    print("[Model] Loading ControlNet...")
    LATENT_CHANNELS = 4
    controlnet = init_controlnet_ct(
        checkpoints_path=None,
        context_dim=int(args.controlnet_context_dim),
        conditioning_embedding_in_channels=LATENT_CHANNELS + 1
    ).to(device)
    if args.cnet_ckpt:
        load_weights_flex(controlnet, args.cnet_ckpt, prefer_ema=bool(args.use_ema), name="ControlNet")
    controlnet.eval()

    # --- Data Loading ---
    from datasets_three_d import LungControlNetPairsCSVDataset
    print(f"[Data] Loading CSV: {args.val_csv}")
    dataset = LungControlNetPairsCSVDataset(
        args.val_csv,
        pairing="baseline_to_all",
        pairs_per_patient_cap=None,
        dt_min=float(args.dt_min),
        dt_max=float(args.dt_max),
        append_mask_token=bool(args.append_mask_token),
        verbose=True,
    )

    # --- Scheduler ---
    ddim = DDIMScheduler(
        num_train_timesteps=1000, schedule="scaled_linear_beta",
        beta_start=0.0015, beta_end=0.0205, clip_sample=False, prediction_type="epsilon"
    )
    ddim.set_timesteps(int(args.num_inference_steps))

    sf = float(args.scale_factor)
    safe_clamp = float(args.safe_clamp)

    os.makedirs(args.output_dir, exist_ok=True)
    total = len(dataset)
    saved = 0

    # Access raw dataframe for metadata
    df_raw = dataset.df

    print(f"[Inference] Iterating {total} pairs...")

    for idx in range(total):
        sample = dataset[idx]

        # 1. Get Metadata
        i_idx, j_idx = dataset.pairs[idx]
        ri = df_raw.iloc[int(i_idx)]
        rj = df_raw.iloc[int(j_idx)]

        patient_raw = str(ri[dataset.patient_col])
        patient_id = _sanitize_for_path(patient_raw)
        t0 = float(ri[dataset.time_col])
        t1 = float(rj[dataset.time_col])
        dt_val = float(sample["starting_time"].item()) if torch.is_tensor(sample["starting_time"]) else float(sample["starting_time"])

        # 2. Get Geometry
        mask_path = str(rj[args.mask_col]) if args.mask_col in rj else ""
        ref_img_path = resolve_ref_img_path(mask_path)
        affine, header = _get_affine_and_header(ref_img_path, default_spacing=(1.35, 1.35, 1.35))

        # 3. Prepare paths
        patient_dir = os.path.join(args.output_dir, patient_id)
        pair_dir = os.path.join(patient_dir, f"t{t0:g}_to_t{t1:g}_dt{dt_val:.2f}")
        os.makedirs(pair_dir, exist_ok=True)

        # 4. Prepare Tensors
        z_start = sample["starting_latent"].unsqueeze(0).to(device).float() * sf
        z_gt = sample["followup_latent"].unsqueeze(0).to(device).float() * sf
        dt_tens = torch.tensor([dt_val], device=device).float().view(1)

        ctx = sample["context"].unsqueeze(0).to(device).float()
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(1)

        if "diffusion_context" in sample:
            diff_ctx = sample["diffusion_context"].unsqueeze(0).to(device).float()
            if diff_ctx.ndim == 2:
                diff_ctx = diff_ctx.unsqueeze(1)
        else:
            diff_ctx = ctx[..., :int(args.diffusion_context_dim)]

        time_map = dt_tens.view(1, 1, 1, 1, 1).expand(1, 1, *z_start.shape[-3:])
        control_cond = torch.cat([z_start, time_map], dim=1).float()

        # 5. Sampling
        def sample_latent(seed_value: int) -> torch.Tensor:
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)

            z_current = torch.randn_like(z_gt)

            for t in ddim.timesteps:
                t_batch = torch.tensor([t], device=device).long()

                with torch.cuda.amp.autocast(enabled=False):
                    if args.control_scale > 0.0:
                        down, mid = controlnet(
                            x=z_current,
                            timesteps=t_batch,
                            context=ctx,
                            controlnet_cond=control_cond
                        )
                        down = [r * float(args.control_scale) for r in down]
                        mid = mid * float(args.control_scale)
                    else:
                        down, mid = None, None

                    noise_pred_cond = diffusion(
                        x=z_current,
                        timesteps=t_batch,
                        context=diff_ctx,
                        down_block_additional_residuals=down,
                        mid_block_additional_residual=mid
                    )

                    if float(args.guidance_scale) > 1.0:
                        noise_pred_uncond = diffusion(
                            x=z_current,
                            timesteps=t_batch,
                            context=torch.zeros_like(diff_ctx),
                            down_block_additional_residuals=None,
                            mid_block_additional_residual=None
                        )
                        noise_pred = noise_pred_uncond + float(args.guidance_scale) * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                z_current, _ = ddim.step(noise_pred, int(t), z_current)

                if args.per_step_clamp:
                    z_current = z_current.clamp(-safe_clamp, safe_clamp)

            return (z_current / sf).clamp(-7, 7)

        n_avg = max(1, int(args.latent_average_samples))
        z_pred_sum = None
        for sample_idx in range(n_avg):
            seed_value = int(args.seed) + int(idx) * n_avg + sample_idx
            z_pred_i = sample_latent(seed_value)
            z_pred_sum = z_pred_i if z_pred_sum is None else z_pred_sum + z_pred_i

        # 6. Decode & Save
        z_pred_raw = (z_pred_sum / float(n_avg)).clamp(-7, 7)
        vol_pred_hu = decoded_minus1_1_to_hu(_decode_latent_patchwise_robust(ae, z_pred_raw)[0, 0])

        base_name = f"{patient_id}_t{t0:g}_t{t1:g}"
        save_nii_gz(
            vol_pred_hu, affine, header,
            os.path.join(pair_dir, f"{base_name}_PRED.nii.gz"),
            save_int16=args.save_int16
        )

        if args.save_start_gt:
            z_start_raw = (z_start / sf).clamp(-7, 7)
            z_gt_raw = (z_gt / sf).clamp(-7, 7)

            vol_start_hu = decoded_minus1_1_to_hu(_decode_latent_patchwise_robust(ae, z_start_raw)[0, 0])
            vol_gt_hu = decoded_minus1_1_to_hu(_decode_latent_patchwise_robust(ae, z_gt_raw)[0, 0])

            save_nii_gz(
                vol_start_hu, affine, header,
                os.path.join(pair_dir, f"{base_name}_START.nii.gz"),
                save_int16=args.save_int16
            )
            save_nii_gz(
                vol_gt_hu, affine, header,
                os.path.join(pair_dir, f"{base_name}_GT.nii.gz"),
                save_int16=args.save_int16
            )

        saved += 1
        if saved % 10 == 0:
            print(f"[Save] {saved}/{total}")

    print(f"[Done] Saved {saved} pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--diff_ckpt", type=str, required=True)
    parser.add_argument("--cnet_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    parser.add_argument("--mask_col", type=str, default="mask_path", help="CSV col for mask path (used to derive geometry)")

    # Dimensions
    parser.add_argument("--diffusion_context_dim", type=int, default=7)
    parser.add_argument("--controlnet_context_dim", type=int, default=8)

    # Inference knobs
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=300)
    parser.add_argument("--latent_average_samples", type=int, default=5,
                        help="Number of stochastic latent predictions to average before decoding.")
    parser.add_argument("--safe_clamp", type=float, default=2.0391)
    parser.add_argument("--scale_factor", type=float, default=0.2913)

    # Flags
    parser.add_argument("--use_ema", type=str2bool, default=True)
    parser.add_argument("--per_step_clamp", type=str2bool, default=True)
    parser.add_argument("--save_start_gt", type=str2bool, default=True, help="Also save baseline and follow-up reference volumes when present in the CSV.")
    parser.add_argument("--save_int16", action="store_true")

    # Settings
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")

    # Dataset args
    parser.add_argument("--dt_min", type=float, default=0.0)
    parser.add_argument("--dt_max", type=float, default=3.0)
    parser.add_argument("--append_mask_token", type=str2bool, default=True)

    args = parser.parse_args()
    main(args)
