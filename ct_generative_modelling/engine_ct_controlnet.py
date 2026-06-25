# engine_ct_controlnet.py
#
# ControlNet engine:
#  - warmup + cosine LR schedule (per microstep)
#  - gradient accumulation + DDP no_sync
#  - EMA update hook
#  - Option A context split:
#       diffusion_context = batch["diffusion_context"] if present else slice(context[..., :diff_ctx_dim])
#       controlnet_context = batch["context"]
#  - CFG dropout (drops CONTROLNET context; diffusion context is dropped too to keep them consistent per-sample)
#  - optional lung mask voxel weighting (latent-space mask)
#  - includes DDIM visualization helper for paired sampling
#
# Batch keys expected:
#   starting_latent, followup_latent, starting_time (dt_norm),
#   context (controlnet context), optional diffusion_context
#   optional mask keys: followup_mask/mask/lung_mask/starting_mask

import math
import os
from contextlib import nullcontext
from typing import Iterable, Dict, Optional, Tuple

import numpy as np
import torch

from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from util.misc import MetricLogger, SmoothedValue
from util.misc import NativeScalerWithGradNormCount as NativeScaler


# ----------------------------
# LR schedule (warmup + cosine)
# ----------------------------
def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch_float: float, args) -> float:
    """
    Warmup for args.warmup_epochs, then cosine decay to args.min_lr until args.epochs.
    Uses args.lr as the MAX LR.
    Returns current lr (group 0).
    """
    max_lr = float(getattr(args, "lr", 1e-4))
    min_lr = float(getattr(args, "min_lr", 0.0))
    warmup_epochs = float(getattr(args, "warmup_epochs", 0))
    total_epochs = float(getattr(args, "epochs", 1))

    if warmup_epochs > 0 and epoch_float < warmup_epochs:
        lr = max_lr * (epoch_float / warmup_epochs)
    else:
        if total_epochs <= warmup_epochs:
            lr = min_lr
        else:
            # progress in [0,1]
            t = (epoch_float - warmup_epochs) / (total_epochs - warmup_epochs)
            t = min(max(t, 0.0), 1.0)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ----------------------------
# helpers
# ----------------------------
def _ensure_context_shape(context: torch.Tensor) -> torch.Tensor:
    """
    Ensure context is (B,1,ctx_dim).
    Accepts:
      - (B, ctx_dim)
      - (B, 1, ctx_dim)
    """
    if context.ndim == 2:
        context = context.unsqueeze(1)
    if context.ndim != 3:
        raise ValueError(f"Context must be rank-2 or rank-3. Got shape {tuple(context.shape)}")
    return context


def _ensure_scalar_per_sample(x: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
    """
    Ensure x is scalar per sample -> returns (B,)
    Accepts (B,), (B,1), (B,1,1), etc.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device)
    x = x.to(device).float()
    x = x.view(B, -1)
    if x.shape[1] != 1:
        raise ValueError(f"Scalar must be 1 value per sample. Got shape {tuple(x.shape)}")
    return x[:, 0]  # (B,)


def _get_lung_mask_from_batch(batch: dict) -> Optional[torch.Tensor]:
    """
    Try common keys for latent-space lung masks.
    Returns tensor (B,1,D,H,W) or None.
    """
    for k in ["followup_mask", "mask", "lung_mask", "starting_mask"]:
        if k in batch:
            m = batch[k]
            if not torch.is_tensor(m):
                m = torch.as_tensor(m)
            # possible shapes: (B,1,D,H,W), (B,D,H,W), (D,H,W), (1,D,H,W)
            if m.ndim == 4:
                # assume (B,D,H,W) -> (B,1,D,H,W)
                m = m.unsqueeze(1)
            elif m.ndim == 3:
                # assume (D,H,W) -> (1,1,D,H,W)
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 5:
                pass
            else:
                raise ValueError(f"Unexpected mask shape for key={k}: {tuple(m.shape)}")
            return m
    return None


def _make_voxel_weight_from_mask(mask: torch.Tensor, latents: torch.Tensor, lung_weight: float) -> Optional[torch.Tensor]:
    """
    mask: (B,1,D,H,W)
    latents: (B,C,D,H,W)
    returns weight: (B,1,D,H,W) where inside lung = lung_weight, outside = 1.0
    """
    if mask is None:
        return None
    if mask.shape[0] != latents.shape[0] or mask.shape[-3:] != latents.shape[-3:]:
        return None
    m = (mask.float() > 0.5).float()
    lw = float(lung_weight)
    return 1.0 + (lw - 1.0) * m


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, voxel_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    pred/target: (B,C,D,H,W)
    voxel_weight: (B,1,D,H,W) -> broadcast to channels
    """
    sq = (pred - target) ** 2
    if voxel_weight is not None:
        if voxel_weight.ndim == 4:
            voxel_weight = voxel_weight.unsqueeze(1)
        if voxel_weight.shape[1] == 1 and sq.shape[1] != 1:
            voxel_weight = voxel_weight.expand(-1, sq.shape[1], -1, -1, -1)
        sq = sq * voxel_weight
    return sq.mean()


def _split_contexts(
    batch: dict,
    device: torch.device,
    diffusion_context_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      ctrl_ctx: (B,1,ctrl_dim)  - ALWAYS from batch["context"] (ControlNet context)
      diff_ctx: (B,1,diff_dim)  - from batch["diffusion_context"] if present,
                                else slice ctrl_ctx[..., :diffusion_context_dim]
    """
    if "context" not in batch:
        raise KeyError("Batch missing required key 'context' (ControlNet context).")

    ctrl_ctx = _ensure_context_shape(batch["context"].to(device).float())

    if "diffusion_context" in batch and batch["diffusion_context"] is not None:
        diff_ctx = _ensure_context_shape(batch["diffusion_context"].to(device).float())
    else:
        if ctrl_ctx.shape[-1] < int(diffusion_context_dim):
            raise ValueError(
                f"Need diffusion_context_dim={diffusion_context_dim} but controlnet context has only "
                f"{ctrl_ctx.shape[-1]} dims. Provide batch['diffusion_context'] or increase context dim."
            )
        diff_ctx = ctrl_ctx[..., :int(diffusion_context_dim)].contiguous()

    if diff_ctx.shape[-1] != int(diffusion_context_dim):
        raise ValueError(f"diffusion_context has dim={diff_ctx.shape[-1]} but expected {diffusion_context_dim}")

    return ctrl_ctx, diff_ctx


# ----------------------------
# TRAIN
# ----------------------------
def train_one_epoch_controlnet(
    diffusion: torch.nn.Module,
    controlnet: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScaler,
    scheduler: DDPMScheduler,
    scale_factor: torch.Tensor,
    log_writer=None,
    args=None,
    ema=None,
) -> Dict[str, float]:
    """One epoch of ControlNet training."""
    diffusion.eval()  # frozen UNet
    controlnet.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("lr", SmoothedValue(window_size=20, fmt="{value:.6e}"))
    header = f"ControlNet Train Epoch: [{epoch}]"

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16

    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    cond_dropout_prob = float(getattr(args, "cond_dropout_prob", 0.0))
    use_lung_weighting = bool(getattr(args, "use_lung_weighting", False))
    lung_weight = float(getattr(args, "lung_weight", 1.0))
    diff_ctx_dim = int(getattr(args, "diffusion_context_dim", 7))

    global_step = epoch * len(data_loader)
    is_ddp = hasattr(controlnet, "no_sync")

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):

        # LR schedule per microstep (matches diffusion engine)
        if (step % accum_iter) == 0:
            cur_epoch = epoch + step / len(data_loader)
            cur_lr = adjust_learning_rate(optimizer, cur_epoch, args)
        else:
            cur_lr = optimizer.param_groups[0]["lr"]

        # Load latents (scaled)
        starting_z = batch["starting_latent"].to(device).float() * scale_factor
        followup_z = batch["followup_latent"].to(device).float() * scale_factor
        B = starting_z.shape[0]

        # dt_norm scalar -> (B,)
        dt = _ensure_scalar_per_sample(batch["starting_time"], B, device)

        # Split contexts (Option A)
        ctrl_ctx, diff_ctx = _split_contexts(batch, device=device, diffusion_context_dim=diff_ctx_dim)

        # CFG dropout: apply SAME drop mask to BOTH contexts (keeps conditioning consistent)
        if cond_dropout_prob > 0.0:
            drop_mask = (torch.rand(B, 1, 1, device=device) < cond_dropout_prob)
            ctrl_ctx = torch.where(drop_mask, torch.zeros_like(ctrl_ctx), ctrl_ctx)
            diff_ctx = torch.where(drop_mask, torch.zeros_like(diff_ctx), diff_ctx)

        # Optional lung voxel weighting
        voxel_w = None
        if use_lung_weighting:
            m = _get_lung_mask_from_batch(batch)
            if m is not None:
                m = m.to(device)
                voxel_w = _make_voxel_weight_from_mask(m, followup_z, lung_weight=lung_weight)

        # DDPM noise + timestep
        noise = torch.randn_like(followup_z)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()

        # Gradient accumulation + DDP no_sync
        update_grad = ((step + 1) % accum_iter == 0)
        sync_ctx = nullcontext()
        if is_ddp and (not update_grad):
            sync_ctx = controlnet.no_sync()

        with sync_ctx:
            with torch.autocast(device_type=amp_device, dtype=amp_dtype):

                # controlnet_cond: concat(starting_z, time_channel)
                time_channel = dt.view(B, 1, 1, 1, 1).expand(B, 1, *starting_z.shape[-3:])
                controlnet_cond = torch.cat([starting_z, time_channel], dim=1)  # (B,5,D,H,W)

                # Add noise to follow-up target latent
                images_noised = scheduler.add_noise(
                    original_samples=followup_z,
                    noise=noise,
                    timesteps=timesteps,
                )

                # ControlNet residuals (uses controlnet context)
                down_h, mid_h = controlnet(
                    x=images_noised,
                    timesteps=timesteps,
                    context=ctrl_ctx,
                    controlnet_cond=controlnet_cond.float(),
                )

                # Diffusion prediction (uses diffusion context)
                noise_pred = diffusion(
                    x=images_noised,
                    timesteps=timesteps,
                    context=diff_ctx,
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h,
                )

                loss = _weighted_mse(noise_pred.float(), noise.float(), voxel_weight=voxel_w)

            loss_val = float(loss.detach().cpu())
            if not math.isfinite(loss_val):
                print(f"Non-finite loss {loss_val} at step {step}. Skipping.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # scale for accumulation
            loss = loss / float(accum_iter)

            loss_scaler(
                loss,
                optimizer,
                parameters=controlnet.parameters(),
                update_grad=update_grad,
                clip_grad=getattr(args, "clip_grad", 0.7),
            )

        if update_grad:
            if ema is not None:
                ema.update()
            optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss_val, lr=float(cur_lr))

        if log_writer is not None and (step % 20 == 0 or step == len(data_loader) - 1):
            log_writer.add_scalar("train_controlnet/batch_mse", loss_val, global_step)
            log_writer.add_scalar("train_controlnet/lr", float(cur_lr), global_step)

        global_step += 1

    metric_logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ----------------------------
# EVAL
# ----------------------------
@torch.no_grad()
def eval_one_epoch_controlnet(
    diffusion: torch.nn.Module,
    controlnet: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    scheduler: DDPMScheduler,
    scale_factor: torch.Tensor,
    log_writer=None,
    args=None,
) -> Dict[str, float]:
    """Validation epoch."""
    diffusion.eval()
    controlnet.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    header = f"ControlNet Val Epoch: [{epoch}]"

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16

    use_lung_weighting = bool(getattr(args, "use_lung_weighting", False))
    lung_weight = float(getattr(args, "lung_weight", 1.0))
    diff_ctx_dim = int(getattr(args, "diffusion_context_dim", 7))

    global_step = epoch * len(data_loader)

    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):

        starting_z = batch["starting_latent"].to(device).float() * scale_factor
        followup_z = batch["followup_latent"].to(device).float() * scale_factor
        B = starting_z.shape[0]
        dt = _ensure_scalar_per_sample(batch["starting_time"], B, device)

        ctrl_ctx, diff_ctx = _split_contexts(batch, device=device, diffusion_context_dim=diff_ctx_dim)

        voxel_w = None
        if use_lung_weighting:
            m = _get_lung_mask_from_batch(batch)
            if m is not None:
                m = m.to(device)
                voxel_w = _make_voxel_weight_from_mask(m, followup_z, lung_weight=lung_weight)

        noise = torch.randn_like(followup_z)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()

        with torch.autocast(device_type=amp_device, dtype=amp_dtype):

            time_channel = dt.view(B, 1, 1, 1, 1).expand(B, 1, *starting_z.shape[-3:])
            controlnet_cond = torch.cat([starting_z, time_channel], dim=1)

            images_noised = scheduler.add_noise(
                original_samples=followup_z,
                noise=noise,
                timesteps=timesteps,
            )

            down_h, mid_h = controlnet(
                x=images_noised,
                timesteps=timesteps,
                context=ctrl_ctx,
                controlnet_cond=controlnet_cond.float(),
            )

            noise_pred = diffusion(
                x=images_noised,
                timesteps=timesteps,
                context=diff_ctx,
                down_block_additional_residuals=down_h,
                mid_block_additional_residual=mid_h,
            )

            loss = _weighted_mse(noise_pred.float(), noise.float(), voxel_weight=voxel_w)

        loss_val = float(loss.detach().cpu())
        metric_logger.update(loss=loss_val)

        if log_writer is not None and (step % 20 == 0 or step == len(data_loader) - 1):
            log_writer.add_scalar("val_controlnet/batch_mse", loss_val, global_step)

        global_step += 1

    metric_logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ----------------------------
# Visualization helper: conditional DDIM sampling for pairs
# ----------------------------
@torch.no_grad()
def _decode_latent_patchwise(
    ae: torch.nn.Module,
    z_unscaled: torch.Tensor,
    patch_size: int = 24,
    overlap: float = 0.25,
    use_amp: bool = True,
    denom_eps: float = 1e-3,
) -> torch.Tensor:
    device = z_unscaled.device
    _, _, D, H, W = z_unscaled.shape

    stride = int(round(patch_size * (1.0 - overlap)))
    stride = max(1, stride)

    def _starts(full, roi):
        if roi >= full:
            return [0]
        pos = list(range(0, full - roi + 1, stride))
        if pos[-1] != full - roi:
            pos.append(full - roi)
        return pos

    zs = _starts(D, patch_size)
    ys = _starts(H, patch_size)
    xs = _starts(W, patch_size)

    OD, OH, OW = D * 4, H * 4, W * 4
    out = torch.zeros((1, 1, OD, OH, OW), device="cpu", dtype=torch.float32)
    wgt = torch.zeros_like(out)

    inner_ae = ae.module if hasattr(ae, "module") else ae
    model_dtype = next(inner_ae.parameters()).dtype

    zp_probe = z_unscaled[..., :patch_size, :patch_size, :patch_size].to(device, dtype=model_dtype).contiguous()
    with torch.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        yp_probe = inner_ae.ae.decode(zp_probe) if hasattr(inner_ae, "ae") else inner_ae.decode(zp_probe)
        if isinstance(yp_probe, (tuple, list)):
            yp_probe = yp_probe[0]

    pD, pH, pW = yp_probe.shape[2:]
    w_patch = torch.ones((1, 1, pD, pH, pW), device="cpu", dtype=torch.float32)

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                zp = z_unscaled[..., z0:z0 + patch_size, y0:y0 + patch_size, x0:x0 + patch_size]
                zp = zp.to(device, dtype=model_dtype).contiguous()

                with torch.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    yp = inner_ae.ae.decode(zp) if hasattr(inner_ae, "ae") else inner_ae.decode(zp)
                    if isinstance(yp, (tuple, list)):
                        yp = yp[0]

                yp = yp.detach().cpu().float()
                zp0, yp0, xp0 = z0 * 4, y0 * 4, x0 * 4
                curr_pD, curr_pH, curr_pW = yp.shape[2:]

                out[..., zp0:zp0 + curr_pD, yp0:yp0 + curr_pH, xp0:xp0 + curr_pW] += yp
                wgt[..., zp0:zp0 + curr_pD, yp0:yp0 + curr_pH, xp0:xp0 + curr_pW] += w_patch[..., :curr_pD, :curr_pH, :curr_pW]

    return out / wgt.clamp_min(denom_eps)


def _to_uint8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).round().astype(np.uint8)


@torch.no_grad()
def sample_and_save_controlnet_pair_slices(
    *,
    ae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    controlnet_model: torch.nn.Module,
    ddim_scheduler: DDIMScheduler,
    scale_factor: torch.Tensor,
    data_loader: Iterable,
    out_dir: str,
    epoch: int,
    device: torch.device,
    num_pairs: int = 1,
    num_inference_steps: int = 100,
    decode_patch_size: int = 24,
    decode_overlap: float = 0.25,
    use_amp: bool = True,
    seed: int = 123,
    log_writer=None,
    diffusion_context_dim: int = 7,
):
    """
    Visualize start (baseline), GT followup, and predicted followup.
    Saves concatenated (start | gt | pred) slices for axial/coronal/sagittal.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    epoch_dir = os.path.join(out_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    ddim_scheduler.set_timesteps(num_inference_steps=int(num_inference_steps))

    batch = next(iter(data_loader))
    starting_z = batch["starting_latent"].to(device).float() * scale_factor
    followup_z = batch["followup_latent"].to(device).float() * scale_factor
    B = starting_z.shape[0]
    dt = _ensure_scalar_per_sample(batch["starting_time"], B, device)

    ctrl_ctx, diff_ctx = _split_contexts(batch, device=device, diffusion_context_dim=int(diffusion_context_dim))

    n = min(int(num_pairs), B)
    for i in range(n):
        z_start = starting_z[i:i + 1]
        z_gt = followup_z[i:i + 1]
        dt_i = dt[i:i + 1]
        ctrl_i = ctrl_ctx[i:i + 1]
        diff_i = diff_ctx[i:i + 1]

        # Start from pure noise for followup generation
        z = torch.randn_like(z_gt, generator=g)

        current = z
        for t in ddim_scheduler.timesteps:
            t_tensor = torch.tensor([int(t)], device=device, dtype=torch.long)

            time_channel = dt_i.view(1, 1, 1, 1, 1).expand(1, 1, *z_start.shape[-3:])
            controlnet_cond = torch.cat([z_start, time_channel], dim=1)  # (1,5,D,H,W)

            with torch.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                down_h, mid_h = controlnet_model(
                    x=current,
                    timesteps=t_tensor,
                    context=ctrl_i,
                    controlnet_cond=controlnet_cond.float(),
                )
                noise_pred = diffusion_model(
                    x=current,
                    timesteps=t_tensor,
                    context=diff_i,
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h,
                )

            current, _ = ddim_scheduler.step(noise_pred, int(t), current)

        # Unscale for decode
        z_pred_unscaled = (current / scale_factor).clamp(-7, 7)
        z_gt_unscaled = (z_gt / scale_factor).clamp(-7, 7)
        z_start_unscaled = (z_start / scale_factor).clamp(-7, 7)

        pred_vol = _decode_latent_patchwise(
            ae, z_pred_unscaled, patch_size=int(decode_patch_size), overlap=float(decode_overlap), use_amp=bool(use_amp)
        )
        gt_vol = _decode_latent_patchwise(
            ae, z_gt_unscaled, patch_size=int(decode_patch_size), overlap=float(decode_overlap), use_amp=bool(use_amp)
        )
        st_vol = _decode_latent_patchwise(
            ae, z_start_unscaled, patch_size=int(decode_patch_size), overlap=float(decode_overlap), use_amp=bool(use_amp)
        )

        def _to01(v):
            v = v.clamp(-1.0, 1.0)
            return ((v + 1.0) * 0.5).clamp(0.0, 1.0)

        pred01 = _to01(pred_vol)[0, 0].numpy()
        gt01 = _to01(gt_vol)[0, 0].numpy()
        st01 = _to01(st_vol)[0, 0].numpy()

        D, H, W = pred01.shape
        slices = {
            "axial": (D // 2, slice(None), slice(None)),
            "coronal": (slice(None), H // 2, slice(None)),
            "sagittal": (slice(None), slice(None), W // 2),
        }

        for plane, idxs in slices.items():
            pred_u8 = _to_uint8(pred01[idxs])
            gt_u8 = _to_uint8(gt01[idxs])
            st_u8 = _to_uint8(st01[idxs])

            canvas = np.concatenate([st_u8, gt_u8, pred_u8], axis=1)
            out_png = os.path.join(epoch_dir, f"pair_{i:02d}_{plane}.png")
            plt.imsave(out_png, canvas, cmap="gray", vmin=0, vmax=255)

            if log_writer is not None:
                log_writer.add_image(
                    f"vis_controlnet/epoch_{epoch:04d}/pair_{i:02d}_{plane}",
                    torch.from_numpy(canvas).unsqueeze(0),
                    epoch,
                    dataformats="CHW",
                )

    if log_writer is not None:
        log_writer.flush()
