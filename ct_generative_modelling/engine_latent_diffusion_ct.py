# engine_latent_diffusion_ct.py
import math
from typing import Iterable, Dict, Optional
from contextlib import nullcontext

import torch

from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler

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


@torch.no_grad()
def _encode_to_latent(ae, x, use_mean: bool = True):
    mu, sigma = ae.ae.encode(x)
    logvar = 2.0 * torch.log(sigma.clamp_min(1e-6))
    logvar = torch.clamp(logvar, min=-12.0, max=6.0)

    if use_mean:
        return mu

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    voxel_weight: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sq = (pred - target) ** 2

    if voxel_weight is not None:
        if voxel_weight.ndim == 4:
            voxel_weight = voxel_weight.unsqueeze(1)
        if voxel_weight.shape[1] == 1 and sq.shape[1] != 1:
            voxel_weight = voxel_weight.expand(-1, sq.shape[1], -1, -1, -1)
        sq = sq * voxel_weight

    per_sample = sq.mean(dim=(1, 2, 3, 4))

    if sample_weight is not None:
        sample_weight = sample_weight.view(-1)
        per_sample = per_sample * sample_weight

    return per_sample.mean()


def _get_covariate_sample_weight(context: torch.Tensor, args, device) -> torch.Tensor:
    if context is None:
        return torch.ones(1, device=device)
    mask_bit = context[:, 0, -1].float()
    w_present = float(getattr(args, "covariate_present_weight", 1.0))
    return 1.0 + (w_present - 1.0) * mask_bit


def _get_lung_voxel_weight(batch: dict, latents: torch.Tensor, args, device) -> Optional[torch.Tensor]:
    if not getattr(args, "use_lung_weighting", False):
        return None

    if "mask" in batch:
        m = batch["mask"]
    elif "lung_mask" in batch:
        m = batch["lung_mask"]
    else:
        return None

    if not torch.is_tensor(m):
        m = torch.as_tensor(m)
    m = m.to(device).float()

    if m.ndim == 4:
        m = m.unsqueeze(1)

    if m.shape[-3:] != latents.shape[-3:]:
        return None

    lw = float(getattr(args, "lung_weight", 1.0))
    return 1.0 + (lw - 1.0) * m


def train_one_epoch(
    ae: torch.nn.Module,
    diffusion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScaler,
    scheduler: DDPMScheduler,
    inferer: DiffusionInferer,
    scale_factor: torch.Tensor,
    log_writer=None,
    use_latent_npz: bool = False,
    cond_dropout_prob: float = 0.0,
    args=None,
    ema=None,
) -> Dict[str, float]:

    diffusion.train()
    ae.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("lr", SmoothedValue(window_size=20, fmt="{value:.6e}"))
    header = f"LatentDiffusion Train Epoch: [{epoch}]"

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16

    accum_iter = int(getattr(args, "accum_iter", 1))
    accum_iter = max(1, accum_iter)

    global_step = epoch * len(data_loader)

    # Check if model handles no_sync (DDP wrapper)
    is_ddp = hasattr(diffusion, "no_sync")

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):

        # ---- LR schedule per (micro)step like AE
        if (step % accum_iter) == 0:
            cur_epoch = epoch + step / len(data_loader)
            cur_lr = adjust_learning_rate(optimizer, cur_epoch, args)
        else:
            cur_lr = optimizer.param_groups[0]["lr"]

        # ---- Load data
        if use_latent_npz:
            z0 = batch["latent"].to(device).float()

            if args.conditioning_mode == "none":
                context = None
            else:
                context = batch["context"].to(device).float()
                if context.ndim == 2:
                    context = context.unsqueeze(1)
        else:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            x = x * 2.0 - 1.0
            with torch.no_grad():
                z0 = _encode_to_latent(ae, x, use_mean=True)
            context = None

        # ---- Semi CFG dropout
        if args.conditioning_mode == "semi" and context is not None:
            Bc = context.shape[0]
            drop_mask = (torch.rand(Bc, 1, 1, device=device) < float(cond_dropout_prob))
            context = torch.where(drop_mask, torch.zeros_like(context), context)
        elif args.conditioning_mode == "none":
            context = None

        # ---- Diffusion loss
        latents = z0 * scale_factor
        noise = torch.randn_like(latents)
        B = latents.shape[0]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()

        voxel_w = _get_lung_voxel_weight(batch, latents, args, device)
        sample_w = None
        if getattr(args, "use_covariate_weighting", False) and (context is not None):
            sample_w = _get_covariate_sample_weight(context, args, device)

        # -----------------------------------------------------------
        # DDP OPTIMIZATION: Context Manager
        # Only sync grads on the step where we actually update weights
        # -----------------------------------------------------------
        update_grad = ((step + 1) % accum_iter == 0)

        sync_ctx = nullcontext()
        if is_ddp and (not update_grad):
            # If NOT updating, disable gradient sync to save bandwidth
            sync_ctx = diffusion.no_sync()

        with sync_ctx:
            with torch.autocast(device_type=amp_device, dtype=amp_dtype):
                mode = "uncond" if args.conditioning_mode == "none" else "crossattn"
                noise_pred = inferer(
                    inputs=latents,
                    diffusion_model=diffusion,
                    noise=noise,
                    timesteps=timesteps,
                    condition=context if mode == "crossattn" else None,
                    mode=mode,
                )
                # compute loss in fp32 for stability
                loss = _weighted_mse(
                    noise_pred.float(),
                    noise.float(),
                    voxel_weight=voxel_w,
                    sample_weight=sample_w,
                )

            loss_val = float(loss.detach().cpu())
            if not math.isfinite(loss_val):
                print(f"Non-finite loss {loss_val} at step {step}. Skipping.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # ---- Gradient accumulation
            loss = loss / float(accum_iter)

            # Helper handles backward internally
            loss_scaler(
                loss,
                optimizer,
                parameters=diffusion.parameters(),
                update_grad=update_grad,
                clip_grad=getattr(args, "clip_grad", 0.7)
            )

        if update_grad:
            if ema is not None:
                ema.update()
            optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss_val, lr=float(cur_lr))

        if log_writer is not None and (step % 20 == 0 or step == len(data_loader) - 1):
            log_writer.add_scalar("train/batch_mse", loss_val, global_step)
            log_writer.add_scalar("train/lr", float(cur_lr), global_step)

        global_step += 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_one_epoch(
    ae: torch.nn.Module,
    diffusion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    scheduler: DDPMScheduler,
    inferer: DiffusionInferer,
    scale_factor: torch.Tensor,
    log_writer=None,
    use_latent_npz: bool = False,
    args=None
) -> Dict[str, float]:

    diffusion.eval()
    ae.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    header = f"LatentDiffusion Val Epoch: [{epoch}]"

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16

    global_step = epoch * len(data_loader)

    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):

        if use_latent_npz:
            z0 = batch["latent"].to(device).float()
            if args.conditioning_mode == "none":
                context = None
            else:
                context = batch["context"].to(device).float()
                if context.ndim == 2:
                    context = context.unsqueeze(1)
        else:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            x = x * 2.0 - 1.0
            z0 = _encode_to_latent(ae, x, use_mean=True)
            context = None

        if args.conditioning_mode == "none":
            context = None

        latents = z0 * scale_factor
        noise = torch.randn_like(latents)
        B = latents.shape[0]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()

        voxel_w = _get_lung_voxel_weight(batch, latents, args, device)
        sample_w = None
        if getattr(args, "use_covariate_weighting", False) and (context is not None):
            sample_w = _get_covariate_sample_weight(context, args, device)

        with torch.autocast(device_type=amp_device, dtype=amp_dtype):
            mode = "uncond" if args.conditioning_mode == "none" else "crossattn"
            noise_pred = inferer(
                inputs=latents,
                diffusion_model=diffusion,
                noise=noise,
                timesteps=timesteps,
                condition=context if mode == "crossattn" else None,
                mode=mode,
            )

            loss = _weighted_mse(
                noise_pred.float(),
                noise.float(),
                voxel_weight=voxel_w,
                sample_weight=sample_w,
            )

        loss_val = float(loss.detach().cpu())
        metric_logger.update(loss=loss_val)

        if log_writer is not None and (step % 20 == 0 or step == len(data_loader) - 1):
            log_writer.add_scalar("val/batch_mse", loss_val, global_step)

        global_step += 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
