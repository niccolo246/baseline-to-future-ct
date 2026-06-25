# IPF ControlNet adaptation with lung-biased patch-decoded supervision.

import argparse
import datetime
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from torch.utils.tensorboard import SummaryWriter
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from ct_generative_modelling.networks_shallow import (
    init_controlnet_ct,
    init_latent_diffusion_ct,
)
from ct_generative_modelling.engine_ct_controlnet_fine_patch import (
    train_one_epoch_controlnet,
    eval_one_epoch_controlnet,
)

from hybrid_vae_vitconv import ConvAEAdapter, PatchDiscriminator3D


@torch.no_grad()
def _decode_latent_patchwise_robust(ae, z, patch_size=24, overlap=0.25):
    """
    Robust patch-wise decoding that enforces contiguous memory to prevent artifacts.
    """
    device = z.device
    z = z.contiguous()

    _, _, D, H, W = z.shape
    stride = int(round(patch_size * (1.0 - overlap)))
    stride = max(1, stride)

    # Output size (Latent is 64^3, Image is 256^3 -> Factor 4)
    OD, OH, OW = D * 4, H * 4, W * 4

    out = torch.zeros((1, 1, OD, OH, OW), device="cpu", dtype=torch.float32)
    wgt = torch.zeros_like(out)

    model_dtype = next(ae.parameters()).dtype

    def get_starts(dim, psize, stride_):
        starts = list(range(0, dim - psize + 1, stride_))
        if starts[-1] != dim - psize:
            starts.append(dim - psize)
        return starts

    zs = get_starts(D, patch_size, stride)
    ys = get_starts(H, patch_size, stride)
    xs = get_starts(W, patch_size, stride)

    # Probe to get decoded patch shape
    zp_probe = z[..., :patch_size, :patch_size, :patch_size].to(device, dtype=model_dtype)
    with torch.cuda.amp.autocast(enabled=True):
        yp_probe = ae.ae.decode(zp_probe) if hasattr(ae, "ae") else ae.decode(zp_probe)
        if isinstance(yp_probe, (tuple, list)):
            yp_probe = yp_probe[0]

    pD, pH, pW = yp_probe.shape[2:]
    w_patch = torch.ones((1, 1, pD, pH, pW), device="cpu")

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                zp = z[..., z0:z0 + patch_size, y0:y0 + patch_size, x0:x0 + patch_size] \
                    .to(device, dtype=model_dtype).contiguous()

                with torch.cuda.amp.autocast(enabled=True):
                    yp = ae.ae.decode(zp) if hasattr(ae, "ae") else ae.decode(zp)
                    if isinstance(yp, (tuple, list)):
                        yp = yp[0]

                yp = yp.cpu().float()

                zp0, yp0_, xp0 = z0 * 4, y0 * 4, x0 * 4
                curr_pD, curr_pH, curr_pW = yp.shape[2:]

                out[..., zp0:zp0 + curr_pD, yp0_:yp0_ + curr_pH, xp0:xp0 + curr_pW] += yp
                wgt[..., zp0:zp0 + curr_pD, yp0_:yp0_ + curr_pH, xp0:xp0 + curr_pW] += w_patch[..., :curr_pD, :curr_pH, :curr_pW]

    return out / wgt.clamp_min(1e-3)


# -------------------------------------------------------------------------
# 2) Local Visualization Function (Replaces Engine Call)
# -------------------------------------------------------------------------
@torch.no_grad()
def run_visualization_locally(
    ae, diffusion, controlnet,
    loader, scale_factor, device,
    epoch, out_dir, args
):
    """
    Robust visualization that includes:
    1. Control Scale Boosting for crisper anatomy (control_scale).
    2. Relaxed clamping (prevents gray fog).
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Vis] Scanning validation set for high-dt pairs (max pairs: {args.vis_num_pairs})...")

    candidates = []
    max_scan_batches = 15

    dt_min = getattr(loader.dataset, "dt_min", 0.0)
    dt_max = getattr(loader.dataset, "dt_max", 3.0)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_scan_batches:
            break
        dts = batch["starting_time"]
        dts_flat = dts.view(-1).cpu().numpy()
        for i in range(len(dts_flat)):
            candidates.append({"dt": float(dts_flat[i]), "batch": batch, "idx": i})

    candidates.sort(key=lambda x: x["dt"], reverse=True)
    num_vis = min(int(args.vis_num_pairs), len(candidates))
    selected = candidates[:num_vis]

    if len(selected) == 0:
        print("[Vis] Warning: No candidates found. Skipping.")
        return

    vis_ddim = DDIMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0205,
        clip_sample=False,
        prediction_type="epsilon",
    )
    vis_ddim.set_timesteps(int(args.vis_num_inference_steps))

    safe_clamp = 2.0391

    for vis_idx, item in enumerate(selected):
        batch = item["batch"]
        i = item["idx"]
        dt_norm_val = float(item["dt"])

        z_start = batch["starting_latent"][i:i + 1].to(device).float() * scale_factor
        z_gt = batch["followup_latent"][i:i + 1].to(device).float() * scale_factor
        dt_tensor = batch["starting_time"][i:i + 1].to(device).float().view(1)

        ctrl_ctx = batch["context"][i:i + 1].to(device).float()
        if ctrl_ctx.ndim == 2:
            ctrl_ctx = ctrl_ctx.unsqueeze(1)

        if "diffusion_context" in batch:
            diff_ctx = batch["diffusion_context"][i:i + 1].to(device).float()
        else:
            diff_ctx = ctrl_ctx[..., :int(args.diffusion_context_dim)]
        if diff_ctx.ndim == 2:
            diff_ctx = diff_ctx.unsqueeze(1)

        time_map = dt_tensor.view(1, 1, 1, 1, 1).expand(1, 1, *z_start.shape[-3:])
        control_cond = torch.cat([z_start, time_map], dim=1).float()

        z_current = torch.randn_like(z_gt)

        control_scale = 1.0  # knob for checking strength

        for t in vis_ddim.timesteps:
            t_batch = torch.tensor([t], device=device).long()
            with torch.cuda.amp.autocast(enabled=bool(args.vis_use_amp)):
                down_res, mid_res = controlnet(
                    x=z_current,
                    timesteps=t_batch,
                    context=ctrl_ctx,
                    controlnet_cond=control_cond
                )

                down_res = [r * control_scale for r in down_res]
                mid_res = mid_res * control_scale

                noise_pred = diffusion(
                    x=z_current,
                    timesteps=t_batch,
                    context=diff_ctx,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res
                )

            z_current, _ = vis_ddim.step(noise_pred, int(t), z_current)
            z_current = z_current.clamp(-safe_clamp, safe_clamp)

        z_pred_raw = (z_current / scale_factor).clamp(-7.0, 7.0)
        z_start_raw = (z_start / scale_factor).clamp(-7.0, 7.0)
        z_gt_raw = (z_gt / scale_factor).clamp(-7.0, 7.0)

        vol_start = _decode_latent_patchwise_robust(ae, z_start_raw, patch_size=int(args.vis_decode_patch_size), overlap=float(args.vis_decode_overlap))[0, 0]
        vol_gt = _decode_latent_patchwise_robust(ae, z_gt_raw, patch_size=int(args.vis_decode_patch_size), overlap=float(args.vis_decode_overlap))[0, 0]
        vol_pred = _decode_latent_patchwise_robust(ae, z_pred_raw, patch_size=int(args.vis_decode_patch_size), overlap=float(args.vis_decode_overlap))[0, 0]

        def norm_to_u8(v):
            v = torch.clamp(v, -1.0, 1.0)
            v = (v + 1.0) / 2.0
            v = v.cpu().numpy()
            return (v * 255.0).astype(np.uint8)

        vol_start = norm_to_u8(vol_start)
        vol_gt = norm_to_u8(vol_gt)
        vol_pred = norm_to_u8(vol_pred)

        D, H, W = vol_start.shape
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

        axial = np.concatenate([vol_start[mid_d], vol_gt[mid_d], vol_pred[mid_d]], axis=1)
        coronal = np.concatenate([vol_start[:, mid_h], vol_gt[:, mid_h], vol_pred[:, mid_h]], axis=1)
        sagittal = np.concatenate([vol_start[:, :, mid_w], vol_gt[:, :, mid_w], vol_pred[:, :, mid_w]], axis=1)

        real_years = dt_norm_val * (dt_max - dt_min) + dt_min
        fig_name = f"epoch{epoch:03d}_pair{vis_idx}_dt{dt_norm_val:.2f}.png"
        save_path = os.path.join(out_dir, fig_name)

        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        title_str = f"Axial | dt={real_years:.1f}y | Scale={control_scale} | Start->GT->Pred"
        axes[0].imshow(axial, cmap="gray"); axes[0].axis("off"); axes[0].set_title(title_str)
        axes[1].imshow(coronal, cmap="gray"); axes[1].axis("off"); axes[1].set_title("Coronal")
        axes[2].imshow(sagittal, cmap="gray"); axes[2].axis("off"); axes[2].set_title("Sagittal")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"[Vis] Saved {save_path}")


# ----------------------------------------------------------
# EMA Helper Class (Robust Version from Diffusion)
# ----------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_val = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_val.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {
            "decay": float(self.decay),
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state, device):
        if state is None:
            return
        if "decay" in state:
            self.decay = float(state["decay"])

        shadow_state = state.get("shadow", None)
        if shadow_state is None:
            return

        if any(k.startswith("module.") for k in shadow_state.keys()):
            shadow_state = {k.replace("module.", "", 1): v for k, v in shadow_state.items()}

        param_map = dict(self.model.named_parameters())
        missing = []

        for k in self.shadow.keys():
            if k in shadow_state:
                target_dtype = param_map[k].dtype if k in param_map else shadow_state[k].dtype
                self.shadow[k] = shadow_state[k].to(device=device, dtype=target_dtype).clone()
            else:
                missing.append(k)

        if missing:
            print(f"[EMA] WARNING: {len(missing)} keys missing in checkpoint (showing 5): {missing[:5]}")


# ----------------------------
# scale factor estimation
# ----------------------------
@torch.no_grad()
def _estimate_scale_factor_from_pairs(dataset, device: torch.device, num_samples: int = 200) -> torch.Tensor:
    print(f"[scale_factor] Estimating from first {num_samples} samples...")
    stds = []
    n = min(int(num_samples), len(dataset))
    if n <= 0:
        raise ValueError("Cannot estimate scale_factor: dataset is empty.")
    for i in range(n):
        s = dataset[i]
        z = s["followup_latent"]
        z = torch.as_tensor(z, device=device, dtype=torch.float32)
        if z.ndim == 4:
            z = z.unsqueeze(0)
        stds.append(torch.std(z))
    avg_std = torch.stack(stds).mean().clamp_min(1e-8)
    sf = (1.0 / avg_std).to(device)
    print(f"[scale_factor] avg_std={avg_std.item():.6f} -> scale_factor={sf.item():.6f}")
    return sf


def _strip_module_prefix(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


# ----------------------------
# Resume helpers
# ----------------------------
def _load_controlnet_checkpoint(
    *,
    resume_path: str,
    controlnet_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: NativeScaler,
    device: torch.device,
    args,
    scale_factor: torch.Tensor,
    ema=None,
    discriminator_without_ddp: Optional[torch.nn.Module] = None,
    disc_optimizer: Optional[torch.optim.Optimizer] = None,
):
    if not resume_path or (not os.path.isfile(resume_path)):
        return 0, scale_factor

    print(f"[resume] Loading ControlNet from: {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

    # ---- ControlNet weights (EMA-aware) ----
    sd = None
    if bool(args.init_from_ema) and ("ema_state" in ckpt):
        print("[resume] --init_from_ema set. Loading from 'ema_state' shadow...")
        raw_ema = ckpt["ema_state"]
        sd = raw_ema["shadow"] if isinstance(raw_ema, dict) and ("shadow" in raw_ema) else raw_ema
    elif "model" in ckpt:
        if bool(args.init_from_ema) and (ckpt.get("is_ema", False) is False) and ("ema_state" not in ckpt):
            print("[resume] WARNING: --init_from_ema requested but ckpt lacks ema_state; loading raw model.")
        sd = ckpt["model"]
    else:
        sd = ckpt

    sd = _strip_module_prefix(sd)
    missing, unexpected = controlnet_without_ddp.load_state_dict(sd, strict=False)
    print("[resume] Loaded ControlNet weights.")
    if len(missing) > 0:
        print(f"[resume] WARNING: Missing keys (showing first 5): {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[resume] WARNING: Unexpected keys (showing first 5): {unexpected[:5]}")

    # ---- Optimizer / scaler ----
    if optimizer is not None and ckpt.get("optimizer", None) is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[resume] Loaded optimizer state.")
        except Exception as e:
            print(f"[resume] WARNING: could not load optimizer state: {e}")

    if loss_scaler is not None and ckpt.get("scaler", None) is not None:
        try:
            loss_scaler.load_state_dict(ckpt["scaler"])
            print("[resume] Loaded AMP scaler state.")
        except Exception as e:
            print(f"[resume] WARNING: could not load scaler state: {e}")

    # ---- Scale factor ----
    if args.scale_factor is None:
        sf = ckpt.get("scale_factor", None)
        if sf is not None:
            scale_factor = sf.to(device)
            print(f"[resume] Loaded scale_factor from ckpt: {scale_factor.item():.6f}")

    # ---- EMA ----
    if ema is not None:
        if "ema_state" in ckpt:
            print("[resume] Loading EMA state...")
            ema.load_state_dict(ckpt["ema_state"], device)
        else:
            print("[resume] WARNING: EMA enabled but checkpoint lacks 'ema_state'. Re-registering EMA.")
            ema.register()

    # ---- Discriminator (optional) ----
    if discriminator_without_ddp is not None and ("discriminator" in ckpt):
        try:
            disc_sd = _strip_module_prefix(ckpt["discriminator"])
            discriminator_without_ddp.load_state_dict(disc_sd, strict=True)
            print("[resume] Loaded discriminator weights.")
        except Exception as e:
            print(f"[resume] WARNING: could not load discriminator weights: {e}")

    if disc_optimizer is not None and ("disc_optimizer" in ckpt):
        try:
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
            print("[resume] Loaded discriminator optimizer state.")
        except Exception as e:
            print(f"[resume] WARNING: could not load discriminator optimizer state: {e}")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"[resume] start_epoch = {start_epoch}")

    del ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return start_epoch, scale_factor


# ----------------------------
# Args
# ----------------------------
def get_args_parser():
    parser = argparse.ArgumentParser("CT ControlNet training", add_help=False)

    # Data
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--dataset_impl", type=str, default="pairs_csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=6)

    # Conditioning dims
    parser.add_argument("--diffusion_context_dim", type=int, default=7)
    parser.add_argument("--controlnet_context_dim", type=int, default=None)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.05)

    # Checkpoints
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--diff_ckpt", type=str, required=True)
    parser.add_argument("--cnet_ckpt_init", type=str, required=True)
    parser.add_argument("--init_from_ema", type=misc.str2bool, default=True, help="If True, try to load EMA weights from the checkpoint instead of raw weights.")
    parser.add_argument("--resume", type=str, default=None)

    # Optim
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--min_lr", type=float, default=8e-7)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--accum_iter", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Scale factor
    parser.add_argument("--scale_factor", type=float, default=0.2913)
    parser.add_argument("--scale_factor_num_samples", type=int, default=200)

    # Loss weighting
    parser.add_argument("--use_lung_weighting", type=misc.str2bool, default=True)
    parser.add_argument("--lung_weight", type=float, default=1.1)

    # Additional latent losses
    parser.add_argument("--delta_loss_weight", type=float, default=0.01,
                        help="Weight for progression (delta) loss in latent space (try 0.05-0.2).")
    parser.add_argument("--edge_loss_weight", type=float, default=0,
                        help="Weight for 3D gradient/texture loss in latent space (try 0.005-0.05).")

    # EMA
    parser.add_argument("--use_ema", type=misc.str2bool, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--save_ema_every", type=int, default=20)

    # Logging / Output
    parser.add_argument("--output_dir", type=str, default="outputs/controlnet_ipf")
    parser.add_argument("--log_dir", type=str, default="outputs/controlnet_ipf")
    parser.add_argument("--save_every", type=int, default=1)

    # Visualization
    parser.add_argument("--do_vis_sampling", type=misc.str2bool, default=True)
    parser.add_argument("--vis_every", type=int, default=1)
    parser.add_argument("--vis_num_pairs", type=int, default=2)
    parser.add_argument("--vis_num_inference_steps", type=int, default=300)
    parser.add_argument("--vis_seed", type=int, default=123)
    parser.add_argument("--vis_decode_patch_size", type=int, default=24)
    parser.add_argument("--vis_decode_overlap", type=float, default=0.25)
    parser.add_argument("--vis_use_amp", type=misc.str2bool, default=True)
    parser.add_argument("--vis_use_ema", type=misc.str2bool, default=True)

    # Misc
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    parser.add_argument("--clip_grad", type=float, default=0.5, help="Max gradient norm to prevent spikes.")

    # Patch voxel losses (decoded)
    parser.add_argument("--use_patch_voxel_loss", type=misc.str2bool, default=True,
                        help="Enable patchwise decoded voxel losses (L1/grad) using AE decoder.")
    parser.add_argument("--patches_per_volume", type=int, default=3)
    parser.add_argument("--latent_patch_size", type=int, default=24)
    parser.add_argument("--patch_lung_bias_prob", type=float, default=0.8)
    parser.add_argument("--patch_outside_lung_weight", type=float, default=0.7)
    parser.add_argument("--voxel_l1_weight", type=float, default=0.025)
    parser.add_argument("--voxel_grad_weight", type=float, default=0.005)

    # PatchGAN on decoded patches
    parser.add_argument("--use_patch_gan", action="store_true", help="Enable the optional adversarial term; proposed model keeps this disabled.")
    parser.add_argument("--gan_weight", type=float, default=0.0, help="Generator adversarial weight (hinge); proposed model uses 0.")
    parser.add_argument("--gan_lr", type=float, default=1e-4, help="Discriminator LR.")
    parser.add_argument("--gan_start_epoch", type=int, default=8, help="Epoch to start GAN.")
    parser.add_argument("--gan_ramp_epochs", type=int, default=3, help="Ramp GAN weight over these epochs.")
    parser.add_argument("--gan_mode", type=str, default="absolute", choices=["residual", "absolute"],
                        help="Discriminator sees residual (followup-start) or absolute followup.")
    parser.add_argument("--gan_every", type=int, default=1, help="Update GAN losses every N steps.")
    parser.add_argument("--gan_base_channels", type=int, default=64)
    parser.add_argument("--gan_n_layers", type=int, default=3)

    # Oversampling is intended for single-GPU runs.
    parser.add_argument("--oversample_dt", type=misc.str2bool, default=False,
                        help="Oversample large dt pairs via WeightedRandomSampler (single GPU).")
    parser.add_argument("--oversample_dt_strength", type=float, default=3.0,
                        help="How strongly to upweight large dt. 0 disables effect.")
    parser.add_argument("--oversample_dt_power", type=float, default=2.0,
                        help="Nonlinearity for dt weighting. 1=linear, 2=quadratic.")
    parser.add_argument("--oversample_dt_min", type=float, default=0.0,
                        help="Ignore dt below this (still sampled but not upweighted). In [0,1].")
    parser.add_argument("--oversample_dt_max", type=float, default=1.0,
                        help="Cap dt for weighting. In [0,1].")


    # Blurred voxel L1 on decoded patches (shift-tolerant supervision)
    parser.add_argument("--use_voxel_blur_l1", type=misc.str2bool, default=True,
                        help="Enable blurred L1 on decoded voxel patches (misregistration-robust).")
    parser.add_argument("--voxel_blur_l1_weight", type=float, default=0.02,
                        help="Weight for blurred voxel L1 loss (decoded patches).")

    parser.add_argument("--voxel_blur_sigma", type=float, default=1.5,
                        help="Gaussian sigma in voxel space for blurred loss (e.g., 1.0-2.0).")
    parser.add_argument("--voxel_blur_downsample", type=int, default=1,
                        help="Optional speedup: downsample factor before blur (1=off, 2=half-res).")

    # Lung mask + HU window focus (applied to blurred loss weights)
    parser.add_argument("--blur_use_lung_mask", type=misc.str2bool, default=False,
                        help="If True, weight blurred loss mostly inside lung mask (if provided by dataset).")
    parser.add_argument("--blur_hu_min", type=float, default=-1000.0,
                        help="HU min for focus window (e.g. vessels/soft-tissue).")
    parser.add_argument("--blur_hu_max", type=float, default=-300.0,
                        help="HU max for focus window.")
    parser.add_argument("--blur_hu_softness", type=float, default=50.0,
                        help="Softness (HU) for smooth gating; 30-80 works well.")
    parser.add_argument("--blur_focus_weight", type=float, default=0.0,
                        help="Extra multiplier strength inside HU window (0 disables HU focusing).")

    # Constants for HU mapping.
    parser.add_argument("--ct_hu_min", type=float, default=-1200.0)
    parser.add_argument("--ct_hu_max", type=float, default=800.0)


    parser.add_argument(
        "--identity_prob",
        type=float,
        default=0.0,
        help="Optional probability of identity/no-change pairs: forces dt=0 and followup_latent=starting_latent."
    )


    return parser



# ----------------------------
# Main
# ----------------------------
def main(args):

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True


    misc.init_distributed_mode(args)
    print("Args:\n", "{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Repro
    seed = int(args.seed) + misc.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ----------------------------
    # Dataset
    # ----------------------------
    if args.dataset_impl != "pairs_csv":
        raise ValueError("This main expects dataset_impl='pairs_csv'")

    from datasets_three_d import LungControlNetPairsCSVDataset

    dataset_train = LungControlNetPairsCSVDataset(args.train_csv, identity_prob=args.identity_prob)
    dataset_val = LungControlNetPairsCSVDataset(args.val_csv) if args.val_csv else None

    if misc.is_main_process():
        print(f"[Context] diffusion_context_dim (arg) = {int(args.diffusion_context_dim)}")

    if args.controlnet_context_dim is None:
        s0 = dataset_train[0]
        ctx0 = s0["context"]
        args.controlnet_context_dim = int(ctx0.shape[-1])
        if misc.is_main_process():
            print(f"[Context] inferred controlnet_context_dim = {args.controlnet_context_dim}")

    # ----------------------------
    # Loaders
    # ----------------------------
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        ) if dataset_val is not None else None

    else:
        #sampler_train = torch.utils.data.RandomSampler(dataset_train)
        if bool(getattr(args, "oversample_dt", False)):
            from torch.utils.data import WeightedRandomSampler

            print("[Oversample] Building dt-based weights for pairs...")
            dts = np.zeros(len(dataset_train), dtype=np.float32)

            # Fast path: compute dt_norm directly from stored pair indices + df
            # (avoids loading .npy latents)
            for k, (i_df, j_df) in enumerate(dataset_train.pairs):
                ti = float(dataset_train.df.iloc[int(i_df)][dataset_train.time_col])
                tj = float(dataset_train.df.iloc[int(j_df)][dataset_train.time_col])
                dt = max(0.0, tj - ti)
                dts[k] = dataset_train._normalize_dt(dt)

            dt_min = float(getattr(args, "oversample_dt_min", 0.0))
            dt_max = float(getattr(args, "oversample_dt_max", 1.0))
            dts_clip = np.clip(dts, dt_min, dt_max)

            strength = float(getattr(args, "oversample_dt_strength", 4.0))
            power = float(getattr(args, "oversample_dt_power", 2.0))

            # Weight formula: 1 + strength * (dt^power)
            w = 1.0 + strength * (dts_clip ** power)

            # convert to torch double (required)
            weights = torch.as_tensor(w, dtype=torch.double)

            sampler_train = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),   # one "epoch" = len(dataset)
                replacement=True,
            )

            print(f"[Oversample] dt stats: mean={dts.mean():.3f}, p50={np.quantile(dts,0.5):.3f}, "
                  f"p90={np.quantile(dts,0.9):.3f}, max={dts.max():.3f}")
            print(f"[Oversample] weights stats: min={w.min():.3f}, mean={w.mean():.3f}, max={w.max():.3f}")
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )

    data_loader_val = None
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )

    # ----------------------------
    # AE (decode only for vis / patch losses)
    # ----------------------------
    ae = ConvAEAdapter(
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        num_res_blocks=(2, 2, 2),
        latent_channels=4,
        attention_levels=(False, False, False),
        norm_num_groups=32,
        use_checkpointing=True,
        use_convtranspose=False,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        output_sigmoid=False,
    ).to(device)

    if args.ae_ckpt and os.path.isfile(args.ae_ckpt):
        print(f"[AE] Loading checkpoint: {args.ae_ckpt}")
        ck = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model_without_ddp" in ck:
            ae.load_state_dict(_strip_module_prefix(ck["model_without_ddp"]), strict=False)
        elif isinstance(ck, dict) and "model" in ck:
            ae.load_state_dict(_strip_module_prefix(ck["model"]), strict=False)
        else:
            ae.load_state_dict(_strip_module_prefix(ck), strict=False)
        print("[AE] Loaded.")
    else:
        print("[AE] WARNING: ae_ckpt not found -> decoding may be meaningless.")

    ae.float().eval()
    for p in ae.parameters():
        p.requires_grad = False

    # ----------------------------
    # Diffusion (frozen)
    # ----------------------------
    args.conditioning_mode = "full"
    diffusion = init_latent_diffusion_ct(
        checkpoints_path=None,
        args=args,
        context_dim=int(args.diffusion_context_dim),
    ).to(device)

    diff_ckpt = torch.load(args.diff_ckpt, map_location="cpu", weights_only=False)
    diffusion.load_state_dict(_strip_module_prefix(diff_ckpt["model"] if "model" in diff_ckpt else diff_ckpt), strict=False)
    print("[Diffusion] Loaded.")

    if args.distributed:
        diffusion = torch.nn.parallel.DistributedDataParallel(
            diffusion, device_ids=[args.gpu], find_unused_parameters=False
        )
        diffusion_without_ddp = diffusion.module
    else:
        diffusion_without_ddp = diffusion

    diffusion_without_ddp.eval()
    for p in diffusion_without_ddp.parameters():
        p.requires_grad = False

    # ----------------------------
    # ControlNet
    # ----------------------------
    LATENT_CHANNELS = 4
    CONTROLNET_COND_CHANNELS = LATENT_CHANNELS + 1

    controlnet = init_controlnet_ct(
        checkpoints_path=None,
        context_dim=int(args.controlnet_context_dim),
        conditioning_embedding_in_channels=int(CONTROLNET_COND_CHANNELS),
    ).to(device)

    if args.cnet_ckpt_init:
        print(f"[ControlNet] Loading init weights from {args.cnet_ckpt_init}...")
        ckpt = torch.load(args.cnet_ckpt_init, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        controlnet.load_state_dict(_strip_module_prefix(sd), strict=False)
    else:
        print("[ControlNet] Smart-copying weights from diffusion...")
        diff_sd = diffusion_without_ddp.state_dict()
        cn_sd = controlnet.state_dict()
        new_sd = {}
        count_copied = 0
        count_skipped = 0
        for k_src, v_src in diff_sd.items():
            k_dest = k_src
            if k_src.startswith("unet."):
                k_dest = k_src.replace("unet.", "controlnet.", 1)
            if k_dest in cn_sd:
                v_dest = cn_sd[k_dest]
                if v_src.shape == v_dest.shape:
                    new_sd[k_dest] = v_src
                    count_copied += 1
                else:
                    if count_skipped < 5:
                        print(f"  [Skip] Shape mismatch for {k_dest}: src={v_src.shape} vs dest={v_dest.shape}")
                    count_skipped += 1
        controlnet.load_state_dict(new_sd, strict=False)
        print(f"[ControlNet] Init complete. Copied {count_copied} layers.")

    if args.distributed:
        controlnet = torch.nn.parallel.DistributedDataParallel(
            controlnet, device_ids=[args.gpu], find_unused_parameters=False
        )
        controlnet_without_ddp = controlnet.module
    else:
        controlnet_without_ddp = controlnet

    # ----------------------------
    # Training Scheduler & Optimizer
    # ----------------------------
    train_ddpm = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0205,
        clip_sample=False,
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(
        controlnet_without_ddp.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    loss_scaler = NativeScaler()

    # ----------------------------
    # Scale factor
    # ----------------------------
    if args.scale_factor is not None:
        scale_factor = torch.tensor(float(args.scale_factor), device=device)
        if misc.is_main_process():
            print(f"[scale_factor] Using provided: {scale_factor.item():.6f}")
    else:
        if misc.is_main_process():
            scale_factor = _estimate_scale_factor_from_pairs(
                dataset_train, device=device, num_samples=int(args.scale_factor_num_samples)
            )
        else:
            scale_factor = torch.tensor(1.0, device=device)
    if args.distributed:
        torch.distributed.broadcast(scale_factor, src=0)

    # ----------------------------
    # Logging
    # ----------------------------
    log_writer = None
    if (not args.distributed or misc.is_main_process()) and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # ----------------------------
    # EMA
    # ----------------------------
    ema = None
    if bool(args.use_ema):
        ema = EMA(controlnet_without_ddp, decay=float(args.ema_decay))
        print(f"[EMA] ControlNet EMA enabled (decay={args.ema_decay})")

    # ----------------------------
    # Optional PatchGAN discriminator.
    # ----------------------------
    discriminator = None
    disc_optimizer = None

    use_patch_gan = bool(getattr(args, "use_patch_gan", False)) and float(getattr(args, "gan_weight", 0.0)) > 0.0
    if use_patch_gan:
        print("[GAN] Initializing PatchDiscriminator3D (decoded patch GAN)")
        discriminator = PatchDiscriminator3D(
            in_channels=1,
            base_channels=int(getattr(args, "gan_base_channels", 64)),
            n_layers=int(getattr(args, "gan_n_layers", 3)),
        ).to(device)

        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(getattr(args, "gan_lr", 1e-4)),
            betas=(0.5, 0.9),
            weight_decay=0.0,
        )

        if args.distributed:
            discriminator = torch.nn.parallel.DistributedDataParallel(
                discriminator, device_ids=[args.gpu], find_unused_parameters=True
            )
            discriminator_without_ddp = discriminator.module
        else:
            discriminator_without_ddp = discriminator
    else:
        discriminator_without_ddp = None

    # Freeze AE + diffusion (safety)
    for p in ae.parameters():
        p.requires_grad = False
    for p in diffusion_without_ddp.parameters():
        p.requires_grad = False

    # ----------------------------
    # Resume training state.
    # ----------------------------
    start_epoch = 0
    if args.resume:
        start_epoch, scale_factor = _load_controlnet_checkpoint(
            resume_path=args.resume,
            controlnet_without_ddp=controlnet_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            device=device,
            args=args,
            scale_factor=scale_factor,
            ema=ema,
            discriminator_without_ddp=discriminator_without_ddp,
            disc_optimizer=disc_optimizer,
        )
        if args.distributed:
            torch.distributed.broadcast(scale_factor, src=0)

    # ----------------------------
    # Loop
    # ----------------------------
    print(f"Start ControlNet training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(start_epoch, int(args.epochs)):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            if data_loader_val:
                data_loader_val.sampler.set_epoch(epoch)

        # Train.
        train_stats = train_one_epoch_controlnet(
            ae=ae,
            diffusion=diffusion,
            controlnet=controlnet,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            scheduler=train_ddpm,
            scale_factor=scale_factor,
            log_writer=log_writer,
            args=args,
            ema=ema,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
        )

        # Val
        val_stats = {}
        if data_loader_val is not None:
            val_stats = eval_one_epoch_controlnet(
                ae=ae,
                diffusion=diffusion,
                controlnet=controlnet,
                data_loader=data_loader_val,
                device=device,
                epoch=epoch,
                scheduler=train_ddpm,
                scale_factor=scale_factor,
                log_writer=log_writer,
                args=args,
            )

        # Visualization (unchanged)
        if bool(args.do_vis_sampling) and misc.is_main_process():
            if (epoch % max(1, int(args.vis_every))) == 0:
                vis_root = os.path.join(args.log_dir if args.log_dir else args.output_dir, "vis_controlnet_pairs")

                use_ema_for_vis = bool(args.vis_use_ema) and (ema is not None)
                if use_ema_for_vis:
                    ema.apply_shadow()
                    vis_root = os.path.join(vis_root, "ema")
                    print("[Vis] Using EMA for vis...")
                else:
                    vis_root = os.path.join(vis_root, "raw")

                try:
                    controlnet_without_ddp.eval()
                    run_visualization_locally(
                        ae=ae,
                        diffusion=diffusion_without_ddp,
                        controlnet=controlnet_without_ddp,
                        loader=data_loader_val if data_loader_val else data_loader_train,
                        scale_factor=scale_factor,
                        device=device,
                        epoch=epoch,
                        out_dir=vis_root,
                        args=args,
                    )
                except Exception as e:
                    print(f"[Vis][Error] {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    controlnet_without_ddp.train()
                    if use_ema_for_vis:
                        ema.restore()

        # ----------------------------
        # Save training state.
        # ----------------------------
        if args.output_dir and misc.is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)

            def _state_dict_maybe_ddp(m):
                if m is None:
                    return None
                return (m.module.state_dict() if hasattr(m, "module") else m.state_dict())

            if ((epoch + 1) % int(args.save_every) == 0) or ((epoch + 1) == int(args.epochs)):
                ckpt_path = os.path.join(args.output_dir, f"controlnet_epoch{epoch}.pth")
                payload = {
                    "epoch": epoch,
                    "model": controlnet_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                    "scale_factor": scale_factor.detach().cpu(),
                    "diffusion_context_dim": int(args.diffusion_context_dim),
                    "controlnet_context_dim": int(args.controlnet_context_dim),
                    "use_lung_weighting": bool(args.use_lung_weighting),
                    "lung_weight": float(args.lung_weight),
                    # GAN
                    "use_patch_gan": bool(use_patch_gan),
                    "gan_weight": float(getattr(args, "gan_weight", 0.0)),
                    "gan_mode": str(getattr(args, "gan_mode", "residual")),
                }
                if ema is not None:
                    payload["ema_state"] = ema.state_dict()
                if discriminator_without_ddp is not None:
                    payload["discriminator"] = _strip_module_prefix(_state_dict_maybe_ddp(discriminator))
                if disc_optimizer is not None:
                    payload["disc_optimizer"] = disc_optimizer.state_dict()

                torch.save(payload, ckpt_path)
                print(f"[epoch {epoch}] Saved -> {ckpt_path}")

            if (ema is not None) and ((epoch + 1) % int(args.save_ema_every) == 0):
                ema_path = os.path.join(args.output_dir, f"controlnet_ema_epoch{epoch}.pth")
                ema.apply_shadow()
                payload = {
                    "epoch": epoch,
                    "model": controlnet_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                    "scale_factor": scale_factor.detach().cpu(),
                    "is_ema": True,
                    "ema_state": ema.state_dict(),
                    # GAN (so ema checkpoints remain runnable)
                    "use_patch_gan": bool(use_patch_gan),
                    "gan_weight": float(getattr(args, "gan_weight", 0.0)),
                    "gan_mode": str(getattr(args, "gan_mode", "residual")),
                }
                if discriminator_without_ddp is not None:
                    payload["discriminator"] = _strip_module_prefix(_state_dict_maybe_ddp(discriminator))
                if disc_optimizer is not None:
                    payload["disc_optimizer"] = disc_optimizer.state_dict()

                torch.save(payload, ema_path)
                ema.restore()
                print(f"[epoch {epoch}] Saved EMA -> {ema_path}")

            # Log stats
            log_stats = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "scale_factor": float(scale_factor.detach().cpu().item()),
                "use_patch_gan": bool(use_patch_gan),
                "gan_weight": float(getattr(args, "gan_weight", 0.0)),
            }
            with open(os.path.join(args.output_dir, "log_controlnet.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    if misc.is_main_process():
        print(f"Training finished in {str(datetime.timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CT ControlNet Training", parents=[get_args_parser()])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)
