# engine_ct_controlnet.py
#
# ControlNet fine-tuning engine:
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
# Latent-space auxiliary losses:
#  - Delta loss (progression): match (x0_pred - start) to (followup - start)
#  - 3D Edge/Texture loss: match finite-difference gradients in D/H/W
#  - Both delta + edge losses can be lung-weighted (same voxel_w logic)
#  - Logs base MSE + aux losses separately
#
# Patchwise decoded voxel losses:
#  - Random latent patch sampling (optionally lung-biased)
#  - Decode only sampled patches through frozen AE decoder (keeps grads to x0_pred)
#  - Voxel L1 loss and voxel gradient loss on decoded patches (sharpness/texture)
#
# Optional PatchGAN integration:
#  - Hinge D loss + hinge G loss on decoded voxel patches (absolute or residual to starting)
#  - D step uses detached real/fake, then G step uses non-detached fake so grads flow to ControlNet
#  - Optional start epoch + linear ramp of gan_weight + update frequency (gan_every)

import math
import os
from contextlib import nullcontext
from typing import Iterable, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

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
            t = (epoch_float - warmup_epochs) / (total_epochs - warmup_epochs)
            t = min(max(t, 0.0), 1.0)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ----------------------------
# PatchGAN hinge losses
# ----------------------------
def _hinge_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    # D wants: real -> +1, fake -> -1
    return (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())


def _hinge_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    # G wants D(fake) to be large/positive
    return (-d_fake).mean()


# ----------------------------
# helpers
# ----------------------------
def _hu_to_norm11(hu: float, hu_min: float = -1200.0, hu_max: float = 800.0) -> float:
    # HU -> [0,1]
    x01 = (hu - hu_min) / max(1e-8, (hu_max - hu_min))
    # [0,1] -> [-1,1]
    return float(x01 * 2.0 - 1.0)

def _hu_width_to_norm11(width_hu: float, hu_min: float = -1200.0, hu_max: float = 800.0) -> float:
    # convert HU delta to delta in [-1,1] scale
    return float((width_hu / max(1e-8, (hu_max - hu_min))) * 2.0)

def _soft_hu_window_weight(x11: torch.Tensor, lo11: float, hi11: float, softness11: float) -> torch.Tensor:
    # x11: (B,1,D,H,W) in [-1,1]
    s = max(float(softness11), 1e-6)
    lo = float(lo11); hi = float(hi11)
    return torch.sigmoid((x11 - lo) / s) * torch.sigmoid((hi - x11) / s)

import torch.nn.functional as F

def _gaussian_kernel_1d(sigma: float, radius: int, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum().clamp_min(1e-12)
    return k

def _gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # x: (B,C,D,H,W)
    if sigma <= 0:
        return x
    radius = int(3 * sigma + 0.5)
    if radius < 1:
        return x

    B, C, D, H, W = x.shape
    k1 = _gaussian_kernel_1d(float(sigma), radius, x.device, x.dtype)  # (K,)

    kD = k1.view(1, 1, -1, 1, 1).expand(C, 1, -1, 1, 1)
    kH = k1.view(1, 1, 1, -1, 1).expand(C, 1, 1, -1, 1)
    kW = k1.view(1, 1, 1, 1, -1).expand(C, 1, 1, 1, -1)

    # depth
    x = F.conv3d(x, kD, padding=(radius, 0, 0), groups=C)
    # height
    x = F.conv3d(x, kH, padding=(0, radius, 0), groups=C)
    # width
    x = F.conv3d(x, kW, padding=(0, 0, radius), groups=C)

    return x



def _maybe_downsample(x: torch.Tensor, ds: int) -> torch.Tensor:
    if ds is None or int(ds) <= 1:
        return x
    ds = int(ds)
    return F.avg_pool3d(x, kernel_size=ds, stride=ds)


def _ensure_context_shape(context: torch.Tensor) -> torch.Tensor:
    """Ensure context is (B,1,ctx_dim). Accepts (B,ctx_dim) or (B,1,ctx_dim)."""
    if context.ndim == 2:
        context = context.unsqueeze(1)
    if context.ndim != 3:
        raise ValueError(f"Context must be rank-2 or rank-3. Got shape {tuple(context.shape)}")
    return context


def _ensure_scalar_per_sample(x: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
    """Ensure x is scalar per sample -> returns (B,). Accepts (B,), (B,1), etc."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device)
    x = x.to(device).float()
    x = x.view(B, -1)
    if x.shape[1] != 1:
        raise ValueError(f"Scalar must be 1 value per sample. Got shape {tuple(x.shape)}")
    return x[:, 0]


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
            if m.ndim == 4:
                m = m.unsqueeze(1)  # (B,D,H,W)->(B,1,D,H,W)
            elif m.ndim == 3:
                m = m.unsqueeze(0).unsqueeze(0)  # (D,H,W)->(1,1,D,H,W)
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


def _split_contexts(batch: dict, device: torch.device, diffusion_context_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      ctrl_ctx: (B,1,ctrl_dim)  - ALWAYS from batch["context"]
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
# x0 (clean latent) estimate + 3D gradient helpers (latent space)
# ----------------------------
def _estimate_x0_from_eps(
    *,
    scheduler: DDPMScheduler,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    eps_pred: torch.Tensor,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> torch.Tensor:
    """
    For epsilon-prediction:
      x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    """
    pred_type = getattr(scheduler, "prediction_type", None)
    if pred_type is not None and str(pred_type).lower() != "epsilon":
        raise ValueError(f"Delta/edge losses assume epsilon prediction, but scheduler.prediction_type={pred_type}")

    alpha_bar = scheduler.alphas_cumprod.to(x_t.device)[timesteps].view(-1, 1, 1, 1, 1)
    one_minus = (1.0 - alpha_bar).clamp(min=0.0)

    x0 = (x_t - one_minus.sqrt() * eps_pred) / alpha_bar.clamp(min=1e-8).sqrt()

    if clamp_min is not None or clamp_max is not None:
        x0 = x0.clamp(
            clamp_min if clamp_min is not None else -float("inf"),
            clamp_max if clamp_max is not None else float("inf"),
        )
    return x0


def _grad3d(z: torch.Tensor):
    gD = z[:, :, 1:, :, :] - z[:, :, :-1, :, :]
    gH = z[:, :, :, 1:, :] - z[:, :, :, :-1, :]
    gW = z[:, :, :, :, 1:] - z[:, :, :, :, :-1]
    return gD, gH, gW


def _expand_weight_to_channels(voxel_w: torch.Tensor, C: int) -> torch.Tensor:
    if voxel_w.ndim == 4:
        voxel_w = voxel_w.unsqueeze(1)  # (B,1,D,H,W)
    if voxel_w.shape[1] == 1 and C != 1:
        voxel_w = voxel_w.expand(-1, C, -1, -1, -1)
    return voxel_w



def _edge_loss_3d(y, x, voxel_w=None):
    def diffs(t):
        dz = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]
        dy = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]
        dx = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]
        return dz, dy, dx

    yz, yy, yx = diffs(y)
    xz, xy, xx = diffs(x)

    if voxel_w is None:
        return ((yz-xz).abs().mean() + (yy-xy).abs().mean() + (yx-xx).abs().mean()) / 3.0

    C = y.shape[1]
    w = voxel_w
    if w.ndim == 4:
        w = w.unsqueeze(1)
    if w.shape[1] == 1 and C != 1:
        w = w.expand(-1, C, -1, -1, -1)

    wD = w[:, :, 1:, :, :]
    wH = w[:, :, :, 1:, :]
    wW = w[:, :, :, :, 1:]

    lossD = ((yz - xz).abs() * wD).mean()
    lossH = ((yy - xy).abs() * wH).mean()
    lossW = ((yx - xx).abs() * wW).mean()
    return (lossD + lossH + lossW) / 3.0



# ----------------------------
# PATCHWISE decoded voxel losses (AE decoder)
# ----------------------------
def _ae_decode(ae: torch.nn.Module, z_unscaled: torch.Tensor) -> torch.Tensor:
    """
    Decode latent (unscaled, typically clamped to [-7,7]) through AE decoder.
    Keeps grads w.r.t. z_unscaled (AE params are frozen so they won't update).
    """
    inner = ae.module if hasattr(ae, "module") else ae
    if hasattr(inner, "ae"):
        y = inner.ae.decode(z_unscaled)
    else:
        y = inner.decode(z_unscaled)
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y


def _repeat_mask_to_voxel(mask_latent: torch.Tensor, up: int = 4) -> torch.Tensor:
    """
    mask_latent: (1,1,ps,ps,ps) -> voxel mask approx (1,1,ps*up,ps*up,ps*up)
    """
    m = mask_latent.float()
    m = m.repeat_interleave(up, dim=2).repeat_interleave(up, dim=3).repeat_interleave(up, dim=4)
    return m


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
    l = (pred - target).abs()
    if w is not None:
        l = l * w
    return l.mean()



def _grad_loss_voxel(pred, target, w=None):
    def _g3(t):
        gD = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]
        gH = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]
        gW = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]
        return gD, gH, gW

    pD, pH, pW = _g3(pred)
    tD, tH, tW = _g3(target)

    if w is None:
        return ((pD - tD).abs().mean()
              + (pH - tH).abs().mean()
              + (pW - tW).abs().mean()) / 3.0

    wD = w[:, :, 1:, :, :]
    wH = w[:, :, :, 1:, :]
    wW = w[:, :, :, :, 1:]

    return (((pD - tD).abs() * wD).mean()
          + ((pH - tH).abs() * wH).mean()
          + ((pW - tW).abs() * wW).mean()) / 3.0



def _sample_latent_patch_boxes(
    mask_latent: Optional[torch.Tensor],
    patch_size: int,
    patches_per_volume: int,
    lung_bias_prob: float,
    B: int,
    D: int,
    H: int,
    W: int,
    device: torch.device,
) -> List[List[Tuple[int, int, int]]]:
    """
    Returns list length B; each element is list of (z0,y0,x0) patch starts.
    mask_latent: (B,1,D,H,W) or None. If present, can bias sampling into lung.
    """
    boxes: List[List[Tuple[int, int, int]]] = []
    ps = int(patch_size)

    max_z = max(D - ps, 0)
    max_y = max(H - ps, 0)
    max_x = max(W - ps, 0)

    for b in range(B):
        b_boxes: List[Tuple[int, int, int]] = []
        m = None
        if mask_latent is not None:
            m = mask_latent[b, 0]  # (D,H,W)

        for _ in range(int(patches_per_volume)):
            use_lung = (m is not None) and (torch.rand((), device=device) < float(lung_bias_prob))

            idx = None
            if use_lung:
                idx = torch.nonzero(m > 0.5, as_tuple=False)
                if idx.numel() == 0:
                    use_lung = False

            if use_lung and idx is not None:
                j = torch.randint(0, idx.shape[0], (1,), device=device).item()
                cz, cy, cx = idx[j].tolist()
            else:
                cz = torch.randint(0, D, (1,), device=device).item()
                cy = torch.randint(0, H, (1,), device=device).item()
                cx = torch.randint(0, W, (1,), device=device).item()

            z0 = int(min(max(cz - ps // 2, 0), max_z))
            y0 = int(min(max(cy - ps // 2, 0), max_y))
            x0 = int(min(max(cx - ps // 2, 0), max_x))

            b_boxes.append((z0, y0, x0))

        boxes.append(b_boxes)

    return boxes

# ----------------------------
# TRAIN
# ----------------------------
def train_one_epoch_controlnet(
    ae: torch.nn.Module,
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
    discriminator=None,
    disc_optimizer=None
) -> Dict[str, float]:
    """
    One epoch of ControlNet training.
    - diffusion gets diffusion_context (NOT controlnet context)
    - controlnet gets controlnet context

    Weighting:
      - loss_mse uses voxel_w     (1 outside lung, lung_weight inside) -> lung-emphasized
      - loss_delta/loss_edge use voxel_w_aux (0 outside, 1 inside)     -> lung-only
      - patch voxel losses: decode sampled patches; optional lung-biased sampling + outside weight

    PatchGAN:
      - uses decoded patches (absolute or residual-to-start) with hinge losses
      - D update uses detached tensors; G update uses non-detached fake so grads reach ControlNet
    """
    diffusion.eval()  # frozen UNet
    controlnet.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_mse", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("lr", SmoothedValue(window_size=20, fmt="{value:.6e}"))

    metric_logger.add_meter("loss_delta", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_edge", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_voxel_l1", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_voxel_grad", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_voxel_blur_l1", SmoothedValue(window_size=20, fmt="{value:.4f}"))

    metric_logger.add_meter("loss_gan_g", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_gan_d", SmoothedValue(window_size=20, fmt="{value:.4f}"))

    header = f"ControlNet Train Epoch: [{epoch}]"

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16

    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    cond_dropout_prob = float(getattr(args, "cond_dropout_prob", 0.0))
    use_lung_weighting = bool(getattr(args, "use_lung_weighting", False))
    lung_weight = float(getattr(args, "lung_weight", 1.0))
    diff_ctx_dim = int(getattr(args, "diffusion_context_dim", 7))

    # Aux loss weights (latent)
    delta_w = float(getattr(args, "delta_loss_weight", 0.0))
    edge_w = float(getattr(args, "edge_loss_weight", 0.0))

    # Patch voxel losses (decoded)
    use_patch_voxel_loss = bool(getattr(args, "use_patch_voxel_loss", False))
    patches_per_volume = int(getattr(args, "patches_per_volume", 1))
    latent_patch_size = int(getattr(args, "latent_patch_size", 24))
    patch_lung_bias_prob = float(getattr(args, "patch_lung_bias_prob", 0.8))
    patch_outside_w = float(getattr(args, "patch_outside_lung_weight", 0.2))
    voxel_l1_w = float(getattr(args, "voxel_l1_weight", 0.0))
    voxel_grad_w = float(getattr(args, "voxel_grad_weight", 0.0))

    # ---- New: blurred voxel L1 (decoded patches, misregistration-robust) ----
    use_voxel_blur_l1 = bool(getattr(args, "use_voxel_blur_l1", False))
    voxel_blur_l1_w = float(getattr(args, "voxel_blur_l1_weight", 0.0))
    blur_sigma = float(getattr(args, "voxel_blur_sigma", 1.5))
    blur_ds = int(getattr(args, "voxel_blur_downsample", 1))

    blur_use_lung_mask = bool(getattr(args, "blur_use_lung_mask", True))
    blur_hu_min = float(getattr(args, "blur_hu_min", -200.0))
    blur_hu_max = float(getattr(args, "blur_hu_max", 300.0))
    blur_hu_soft = float(getattr(args, "blur_hu_softness", 50.0))
    blur_focus_weight = float(getattr(args, "blur_focus_weight", 1.0))

    ct_hu_min = float(getattr(args, "ct_hu_min", -1200.0))
    ct_hu_max = float(getattr(args, "ct_hu_max", 800.0))

    # PatchGAN args
    use_patch_gan = bool(getattr(args, "use_patch_gan", False))
    gan_weight = float(getattr(args, "gan_weight", 0.0))
    gan_start_epoch = int(getattr(args, "gan_start_epoch", 0))
    gan_ramp_epochs = int(getattr(args, "gan_ramp_epochs", 1))
    gan_mode = str(getattr(args, "gan_mode", "residual"))  # "residual" or "absolute"
    gan_every = int(getattr(args, "gan_every", 1))

    gan_enabled = (
        use_patch_gan and
        (gan_weight > 0.0) and
        (discriminator is not None) and
        (disc_optimizer is not None)
    )
    if gan_enabled:
        discriminator.train()

    # Clamp for x0 estimate in SCALED latent units
    try:
        sf_val = float(scale_factor)
    except Exception:
        sf_val = float(scale_factor.item())
    x0_clamp = 7.0 * sf_val

    global_step = epoch * len(data_loader)
    is_ddp = hasattr(controlnet, "no_sync")

    optimizer.zero_grad(set_to_none=True)

    # Ensure scale_factor is a tensor on device
    if not torch.is_tensor(scale_factor):
        scale_factor_t = torch.tensor(float(scale_factor), device=device)
    else:
        scale_factor_t = scale_factor.to(device)
    sf_view = scale_factor_t.reshape(1, 1, 1, 1, 1)

    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):

        # LR schedule per microstep
        if (step % accum_iter) == 0:
            cur_epoch = epoch + step / len(data_loader)
            cur_lr = adjust_learning_rate(optimizer, cur_epoch, args)
        else:
            cur_lr = optimizer.param_groups[0]["lr"]

        # GAN gating + ramp
        gan_active = gan_enabled and (epoch >= gan_start_epoch) and ((step % max(1, gan_every)) == 0)
        if gan_active:
            if gan_ramp_epochs <= 0:
                gan_w_eff = gan_weight
            else:
                frac = float(epoch - gan_start_epoch + 1) / float(gan_ramp_epochs)
                gan_w_eff = gan_weight * min(max(frac, 0.0), 1.0)
        else:
            gan_w_eff = 0.0

        # Load latents (scaled)
        starting_z = batch["starting_latent"].to(device).float() * scale_factor_t
        followup_z = batch["followup_latent"].to(device).float() * scale_factor_t
        B = starting_z.shape[0]

        # dt_norm scalar -> (B,)
        dt = _ensure_scalar_per_sample(batch["starting_time"], B, device)

        # Split contexts (Option A)
        ctrl_ctx, diff_ctx = _split_contexts(batch, device=device, diffusion_context_dim=diff_ctx_dim)

        # CFG dropout: apply SAME drop mask to BOTH contexts
        if cond_dropout_prob > 0.0:
            drop_mask = (torch.rand(B, 1, 1, device=device) < cond_dropout_prob)
            ctrl_ctx = torch.where(drop_mask, torch.zeros_like(ctrl_ctx), ctrl_ctx)
            diff_ctx = torch.where(drop_mask, torch.zeros_like(diff_ctx), diff_ctx)

        # ------------------------------------------------------------
        # Lung weighting
        # ------------------------------------------------------------
        mask_latent = None
        voxel_w = None
        voxel_w_aux = None
        if use_lung_weighting:
            m = _get_lung_mask_from_batch(batch)
            if m is not None:
                mask_latent = m.to(device).float()  # (B,1,D,H,W)
                voxel_w = _make_voxel_weight_from_mask(mask_latent, followup_z, lung_weight=lung_weight)  # emphasized
                voxel_w_aux = (mask_latent > 0.5).float()  # lung-only

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
                controlnet_cond = torch.cat([starting_z, time_channel], dim=1)

                # Add noise to follow-up target latent
                images_noised = scheduler.add_noise(
                    original_samples=followup_z,
                    noise=noise,
                    timesteps=timesteps,
                )

                # ControlNet residuals
                down_h, mid_h = controlnet(
                    x=images_noised,
                    timesteps=timesteps,
                    context=ctrl_ctx,
                    controlnet_cond=controlnet_cond.float(),
                )

                # Diffusion prediction
                noise_pred = diffusion(
                    x=images_noised,
                    timesteps=timesteps,
                    context=diff_ctx,
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h,
                )

                # Base MSE loss on epsilon (lung-emphasized)
                loss_mse = _weighted_mse(noise_pred.float(), noise.float(), voxel_weight=voxel_w)
                loss = loss_mse

                # Aux losses
                loss_delta = torch.zeros((), device=device)
                loss_edge = torch.zeros((), device=device)
                loss_voxel_l1 = torch.zeros((), device=device)
                loss_voxel_grad = torch.zeros((), device=device)
                loss_voxel_blur_l1 = torch.zeros((), device=device)

                loss_gan_g = torch.zeros((), device=device)
                loss_gan_d = torch.zeros((), device=device)

                need_x0 = (
                    (delta_w > 0.0) or
                    (edge_w > 0.0) or
                    (use_patch_voxel_loss and ((voxel_l1_w > 0.0) or (voxel_grad_w > 0.0))) or
                    (use_voxel_blur_l1 and (voxel_blur_l1_w > 0.0)) or
                    (gan_active and gan_w_eff > 0.0)  # need x0_pred to decode fake patches
                )

                x0_pred = None
                if need_x0:
                    x0_pred = _estimate_x0_from_eps(
                        scheduler=scheduler,
                        x_t=images_noised,
                        timesteps=timesteps,
                        eps_pred=noise_pred,
                        clamp_min=-x0_clamp,
                        clamp_max=+x0_clamp,
                    )

                    # Delta/progression (lung-only)
                    if delta_w > 0.0:
                        delta_pred = x0_pred - starting_z
                        delta_gt = followup_z - starting_z
                        loss_delta = _weighted_mse(delta_pred.float(), delta_gt.float(), voxel_weight=voxel_w_aux)
                        loss = loss + delta_w * loss_delta

                    # Edge/texture (latent-space, lung-only)
                    if edge_w > 0.0:
                        loss_edge = _edge_loss_3d(x0_pred.float(), followup_z.float(), voxel_w=voxel_w_aux)
                        loss = loss + edge_w * loss_edge

                    # Patchwise decoded voxel losses and/or GAN and/or blur loss
                    need_patches = (
                        (use_patch_voxel_loss and ((voxel_l1_w > 0.0) or (voxel_grad_w > 0.0))) or
                        (use_voxel_blur_l1 and (voxel_blur_l1_w > 0.0)) or
                        (gan_active and gan_w_eff > 0.0)
                    )

                    if need_patches:
                        _, _, D, H, W = x0_pred.shape
                        boxes = _sample_latent_patch_boxes(
                            mask_latent=mask_latent,
                            patch_size=latent_patch_size,
                            patches_per_volume=patches_per_volume,
                            lung_bias_prob=patch_lung_bias_prob,
                            B=B, D=D, H=H, W=W,
                            device=device,
                        )

                        n_p = 0
                        l1_sum = 0.0
                        g_sum = 0.0
                        blur_sum = 0.0
                        d_sum = 0.0
                        g_adv_sum = 0.0

                        # Precompute HU window params once (python floats)
                        lo11 = _hu_to_norm11(blur_hu_min, ct_hu_min, ct_hu_max)
                        hi11 = _hu_to_norm11(blur_hu_max, ct_hu_min, ct_hu_max)
                        soft11 = _hu_width_to_norm11(blur_hu_soft, ct_hu_min, ct_hu_max)

                        for b in range(B):
                            for (z0, y0, x0) in boxes[b]:
                                z_pred_patch = x0_pred[b:b+1, :, z0:z0+latent_patch_size, y0:y0+latent_patch_size, x0:x0+latent_patch_size]
                                z_gt_patch = followup_z[b:b+1, :, z0:z0+latent_patch_size, y0:y0+latent_patch_size, x0:x0+latent_patch_size]

                                # unscale + clamp for AE decode
                                z_pred_u = (z_pred_patch / sf_view).clamp(-7.0, 7.0)
                                z_gt_u = (z_gt_patch / sf_view).clamp(-7.0, 7.0)

                                v_pred = _ae_decode(ae, z_pred_u)
                                v_gt = _ae_decode(ae, z_gt_u)

                                # weights in voxel-space for this patch (for voxel losses)
                                w_vox = None
                                if mask_latent is not None:
                                    m_patch = mask_latent[b:b+1, :, z0:z0+latent_patch_size, y0:y0+latent_patch_size, x0:x0+latent_patch_size]
                                    m_vox = _repeat_mask_to_voxel(m_patch, up=4)
                                    inside = (m_vox > 0.5).float()
                                    w_vox = patch_outside_w + (1.0 - patch_outside_w) * inside  # (1,1,VD,VH,VW)

                                # Voxel losses (raw)
                                if use_patch_voxel_loss:
                                    if voxel_l1_w > 0.0:
                                        l1_sum = l1_sum + _weighted_l1(v_pred, v_gt, w=w_vox)
                                    if voxel_grad_w > 0.0:
                                        g_sum = g_sum + _grad_loss_voxel(v_pred, v_gt, w=w_vox)

                                # ---- Blurred voxel L1 (shift-tolerant) with lung + HU focus ----
                                if use_voxel_blur_l1 and voxel_blur_l1_w > 0.0:
                                    vp = v_pred.float()
                                    vg = v_gt.float()

                                    vp_ds = _maybe_downsample(vp, blur_ds)
                                    vg_ds = _maybe_downsample(vg, blur_ds)

                                    vp_blur = _gaussian_blur_3d(vp_ds, sigma=blur_sigma)
                                    vg_blur = _gaussian_blur_3d(vg_ds, sigma=blur_sigma)

                                    # start from lung weights (if enabled + available)
                                    if blur_use_lung_mask and (w_vox is not None):
                                        w_focus = _maybe_downsample(w_vox.float(), blur_ds)
                                    else:
                                        w_focus = torch.ones_like(vg_blur[:, :1])

                                    # HU window gate (computed from GT)
                                    if blur_focus_weight > 0.0:
                                        gate = _soft_hu_window_weight(
                                            x11=vg_ds[:, :1],  # GT intensities in [-1,1]
                                            lo11=lo11, hi11=hi11, softness11=soft11
                                        )
                                        w_focus = w_focus * (1.0 + blur_focus_weight * gate)

                                    blur_sum = blur_sum + _weighted_l1(vp_blur, vg_blur, w=w_focus)

                                # PatchGAN losses (hinge)
                                if gan_active and gan_w_eff > 0.0:
                                    # Build discriminator inputs: absolute or residual-to-start
                                    if gan_mode == "residual":
                                        z_st_patch = starting_z[b:b+1, :, z0:z0+latent_patch_size, y0:y0+latent_patch_size, x0:x0+latent_patch_size]
                                        z_st_u = (z_st_patch / sf_view).clamp(-7.0, 7.0)
                                        v_st = _ae_decode(ae, z_st_u)
                                        real_in = (v_gt - v_st)
                                        fake_in = (v_pred - v_st)
                                    else:
                                        real_in = v_gt
                                        fake_in = v_pred

                                    # (A) Discriminator update (detached)
                                    disc_optimizer.zero_grad(set_to_none=True)

                                    d_real = discriminator(real_in.detach())
                                    d_fake = discriminator(fake_in.detach())

                                    d_loss = _hinge_d_loss(d_real, d_fake).float()
                                    d_loss.backward()
                                    disc_optimizer.step()
                                    d_sum = d_sum + d_loss.detach()

                                    # (B) Generator adversarial term (no detach)
                                    d_fake_for_g = discriminator(fake_in)
                                    g_adv = _hinge_g_loss(d_fake_for_g)
                                    g_adv_sum = g_adv_sum + g_adv.detach()

                                    # add to total loss so gradients flow to ControlNet
                                    loss = loss + gan_w_eff * g_adv

                                n_p += 1

                        # finalize patch-based logs/aux terms
                        if n_p > 0:
                            if use_patch_voxel_loss and voxel_l1_w > 0.0:
                                loss_voxel_l1 = l1_sum / float(n_p)
                                loss = loss + voxel_l1_w * loss_voxel_l1
                            if use_patch_voxel_loss and voxel_grad_w > 0.0:
                                loss_voxel_grad = g_sum / float(n_p)
                                loss = loss + voxel_grad_w * loss_voxel_grad

                            if use_voxel_blur_l1 and voxel_blur_l1_w > 0.0:
                                loss_voxel_blur_l1 = blur_sum / float(n_p)
                                loss = loss + voxel_blur_l1_w * loss_voxel_blur_l1

                            if gan_active and gan_w_eff > 0.0:
                                loss_gan_d = d_sum / float(n_p)
                                loss_gan_g = g_adv_sum / float(n_p)

            loss_val = float(loss.detach().cpu())
            if not math.isfinite(loss_val):
                print(f"Non-finite loss {loss_val} at step {step}. Skipping.")
                optimizer.zero_grad(set_to_none=True)
                if gan_active and (disc_optimizer is not None):
                    disc_optimizer.zero_grad(set_to_none=True)
                continue

            # scale for accumulation (ControlNet optimizer only)
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

        metric_logger.update(
            loss=loss_val,
            loss_mse=float(loss_mse.detach().cpu()),
            loss_delta=float(loss_delta.detach().cpu()) if (delta_w > 0.0) else 0.0,
            loss_edge=float(loss_edge.detach().cpu()) if (edge_w > 0.0) else 0.0,
            loss_voxel_l1=float(loss_voxel_l1.detach().cpu()) if (use_patch_voxel_loss and voxel_l1_w > 0.0) else 0.0,
            loss_voxel_grad=float(loss_voxel_grad.detach().cpu()) if (use_patch_voxel_loss and voxel_grad_w > 0.0) else 0.0,
            loss_voxel_blur_l1=float(loss_voxel_blur_l1.detach().cpu()) if (use_voxel_blur_l1 and voxel_blur_l1_w > 0.0) else 0.0,
            loss_gan_g=float(loss_gan_g.detach().cpu()) if (gan_active and gan_w_eff > 0.0) else 0.0,
            loss_gan_d=float(loss_gan_d.detach().cpu()) if (gan_active and gan_w_eff > 0.0) else 0.0,
            lr=float(cur_lr),
        )

        if log_writer is not None and (step % 20 == 0 or step == len(data_loader) - 1):
            log_writer.add_scalar("train_controlnet/loss_total", loss_val, global_step)
            log_writer.add_scalar("train_controlnet/loss_mse", float(loss_mse.detach().cpu()), global_step)
            if delta_w > 0.0:
                log_writer.add_scalar("train_controlnet/loss_delta", float(loss_delta.detach().cpu()), global_step)
            if edge_w > 0.0:
                log_writer.add_scalar("train_controlnet/loss_edge", float(loss_edge.detach().cpu()), global_step)
            if use_patch_voxel_loss and voxel_l1_w > 0.0:
                log_writer.add_scalar("train_controlnet/loss_voxel_l1", float(loss_voxel_l1.detach().cpu()), global_step)
            if use_patch_voxel_loss and voxel_grad_w > 0.0:
                log_writer.add_scalar("train_controlnet/loss_voxel_grad", float(loss_voxel_grad.detach().cpu()), global_step)
            if use_voxel_blur_l1 and voxel_blur_l1_w > 0.0:
                log_writer.add_scalar("train_controlnet/loss_voxel_blur_l1", float(loss_voxel_blur_l1.detach().cpu()), global_step)

            if gan_active and gan_w_eff > 0.0:
                log_writer.add_scalar("train_controlnet/loss_gan_g", float(loss_gan_g.detach().cpu()), global_step)
                log_writer.add_scalar("train_controlnet/loss_gan_d", float(loss_gan_d.detach().cpu()), global_step)
                log_writer.add_scalar("train_controlnet/gan_w_eff", float(gan_w_eff), global_step)

            log_writer.add_scalar("train_controlnet/lr", float(cur_lr), global_step)

        global_step += 1

    metric_logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ----------------------------
# EVAL
# ----------------------------
@torch.no_grad()
def eval_one_epoch_controlnet(
    ae: torch.nn.Module,
    diffusion: torch.nn.Module,
    controlnet: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    scheduler: DDPMScheduler,
    scale_factor: torch.Tensor,
    log_writer=None,
    args=None,
    discriminator=None,
) -> Dict[str, float]:
    """
    Validation epoch.
    Matches TRAIN weighting:
      - loss_mse uses voxel_w     (1 outside lung, lung_weight inside)  -> lung-emphasized
      - loss_delta/loss_edge use voxel_w_aux (0 outside, 1 inside)      -> lung-only

    Also makes eval deterministic (noise + timesteps) via torch.Generator.
    Patch voxel losses are logged for monitoring (still deterministic).

    Note: We DO NOT train the discriminator in eval. If desired, you can log
    discriminator scores (optional) but no optimizer steps here.
    """
    diffusion.eval()
    controlnet.eval()
    if discriminator is not None:
        discriminator.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss_mse", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_delta", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_edge", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_voxel_l1", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_voxel_grad", SmoothedValue(window_size=20, fmt="{value:.4f}"))

    header = f"ControlNet Val Epoch: [{epoch}]"

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16

    use_lung_weighting = bool(getattr(args, "use_lung_weighting", False))
    lung_weight = float(getattr(args, "lung_weight", 1.0))
    diff_ctx_dim = int(getattr(args, "diffusion_context_dim", 7))

    delta_w = float(getattr(args, "delta_loss_weight", 0.0))
    edge_w = float(getattr(args, "edge_loss_weight", 0.0))

    # Patch voxel losses (decoded)
    use_patch_voxel_loss = bool(getattr(args, "use_patch_voxel_loss", False))
    patches_per_volume = int(getattr(args, "patches_per_volume", 1))
    latent_patch_size = int(getattr(args, "latent_patch_size", 24))
    patch_lung_bias_prob = float(getattr(args, "patch_lung_bias_prob", 0.8))
    patch_outside_w = float(getattr(args, "patch_outside_lung_weight", 0.2))
    voxel_l1_w = float(getattr(args, "voxel_l1_weight", 0.0))
    voxel_grad_w = float(getattr(args, "voxel_grad_weight", 0.0))

    # Clamp for x0 estimate in SCALED latent units
    try:
        sf_val = float(scale_factor)
    except Exception:
        sf_val = float(scale_factor.item())
    x0_clamp = 7.0 * sf_val

    # Deterministic eval RNG (DDP-safe)
    base_seed = int(getattr(args, "seed", 0))
    rank = int(getattr(args, "local_rank", 0))
    eval_seed = base_seed + 12345 + 1000 * max(rank, 0) + int(epoch)
    g = torch.Generator(device=device)
    g.manual_seed(eval_seed)

    global_step = epoch * len(data_loader)

    # Ensure scale_factor is a tensor on device
    if not torch.is_tensor(scale_factor):
        scale_factor_t = torch.tensor(float(scale_factor), device=device)
    else:
        scale_factor_t = scale_factor.to(device)
    sf_view = scale_factor_t.reshape(1, 1, 1, 1, 1)

    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):

        starting_z = batch["starting_latent"].to(device).float() * scale_factor_t
        followup_z = batch["followup_latent"].to(device).float() * scale_factor_t
        B = starting_z.shape[0]
        dt = _ensure_scalar_per_sample(batch["starting_time"], B, device)

        ctrl_ctx, diff_ctx = _split_contexts(batch, device=device, diffusion_context_dim=diff_ctx_dim)

        # Lung weighting
        mask_latent = None
        voxel_w = None
        voxel_w_aux = None
        if use_lung_weighting:
            m = _get_lung_mask_from_batch(batch)
            if m is not None:
                mask_latent = m.to(device).float()
                voxel_w = _make_voxel_weight_from_mask(mask_latent, followup_z, lung_weight=lung_weight)
                voxel_w_aux = (mask_latent > 0.5).float()

        # deterministic noise + timesteps
        noise = torch.randn(followup_z.shape, device=device, dtype=followup_z.dtype, generator=g)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device, generator=g).long()

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

            # Base epsilon-MSE (lung-emphasized)
            loss_mse = _weighted_mse(noise_pred.float(), noise.float(), voxel_weight=voxel_w)

            loss_delta = torch.zeros((), device=device)
            loss_edge = torch.zeros((), device=device)
            loss_voxel_l1 = torch.zeros((), device=device)
            loss_voxel_grad = torch.zeros((), device=device)

            need_x0 = (
                (delta_w > 0.0) or
                (edge_w > 0.0) or
                (use_patch_voxel_loss and ((voxel_l1_w > 0.0) or (voxel_grad_w > 0.0)))
            )

            if need_x0:
                x0_pred = _estimate_x0_from_eps(
                    scheduler=scheduler,
                    x_t=images_noised,
                    timesteps=timesteps,
                    eps_pred=noise_pred,
                    clamp_min=-x0_clamp,
                    clamp_max=+x0_clamp,
                )

                # Delta + Edge are lung-only (voxel_w_aux)
                if delta_w > 0.0:
                    delta_pred = x0_pred - starting_z
                    delta_gt = followup_z - starting_z
                    loss_delta = _weighted_mse(delta_pred.float(), delta_gt.float(), voxel_weight=voxel_w_aux)

                if edge_w > 0.0:
                    loss_edge = _edge_loss_3d(x0_pred.float(), followup_z.float(), voxel_w=voxel_w_aux)

                # Patchwise voxel losses (monitor only in eval)
                if use_patch_voxel_loss and ((voxel_l1_w > 0.0) or (voxel_grad_w > 0.0)):
                    _, _, D, H, W = x0_pred.shape
                    boxes = _sample_latent_patch_boxes(
                        mask_latent=mask_latent,
                        patch_size=latent_patch_size,
                        patches_per_volume=patches_per_volume,
                        lung_bias_prob=patch_lung_bias_prob,
                        B=B, D=D, H=H, W=W,
                        device=device,
                    )

                    n_p = 0
                    l1_sum = 0.0
                    g_sum = 0.0

                    for b in range(B):
                        for (z0, y0, x0) in boxes[b]:
                            z_pred_patch = x0_pred[b:b+1, :, z0:z0+latent_patch_size, y0:y0+latent_patch_size, x0:x0+latent_patch_size]
                            z_gt_patch = followup_z[b:b+1, :, z0:z0+latent_patch_size, y0:y0+latent_patch_size, x0:x0+latent_patch_size]

                            z_pred_u = (z_pred_patch / sf_view).clamp(-7.0, 7.0)
                            z_gt_u = (z_gt_patch / sf_view).clamp(-7.0, 7.0)

                            v_pred = _ae_decode(ae, z_pred_u)
                            v_gt = _ae_decode(ae, z_gt_u)

                            w_vox = None
                            if mask_latent is not None:
                                m_patch = mask_latent[b:b+1, :, z0:z0+latent_patch_size, y0:y0+latent_patch_size, x0:x0+latent_patch_size]
                                m_vox = _repeat_mask_to_voxel(m_patch, up=4)
                                inside = (m_vox > 0.5).float()
                                w_vox = patch_outside_w + (1.0 - patch_outside_w) * inside

                            if voxel_l1_w > 0.0:
                                l1_sum = l1_sum + _weighted_l1(v_pred, v_gt, w=w_vox)
                            if voxel_grad_w > 0.0:
                                g_sum = g_sum + _grad_loss_voxel(v_pred, v_gt, w=w_vox)

                            n_p += 1

                    if n_p > 0:
                        if voxel_l1_w > 0.0:
                            loss_voxel_l1 = l1_sum / float(n_p)
                        if voxel_grad_w > 0.0:
                            loss_voxel_grad = g_sum / float(n_p)

        metric_logger.update(
            loss_mse=float(loss_mse.detach().cpu()),
            loss_delta=float(loss_delta.detach().cpu()) if (delta_w > 0.0) else 0.0,
            loss_edge=float(loss_edge.detach().cpu()) if (edge_w > 0.0) else 0.0,
            loss_voxel_l1=float(loss_voxel_l1.detach().cpu()) if (use_patch_voxel_loss and voxel_l1_w > 0.0) else 0.0,
            loss_voxel_grad=float(loss_voxel_grad.detach().cpu()) if (use_patch_voxel_loss and voxel_grad_w > 0.0) else 0.0,
        )

        if log_writer is not None and (step % 20 == 0 or step == len(data_loader) - 1):
            log_writer.add_scalar("val_controlnet/loss_mse", float(loss_mse.detach().cpu()), global_step)
            if delta_w > 0.0:
                log_writer.add_scalar("val_controlnet/loss_delta", float(loss_delta.detach().cpu()), global_step)
            if edge_w > 0.0:
                log_writer.add_scalar("val_controlnet/loss_edge", float(loss_edge.detach().cpu()), global_step)
            if use_patch_voxel_loss and voxel_l1_w > 0.0:
                log_writer.add_scalar("val_controlnet/loss_voxel_l1", float(loss_voxel_l1.detach().cpu()), global_step)
            if use_patch_voxel_loss and voxel_grad_w > 0.0:
                log_writer.add_scalar("val_controlnet/loss_voxel_grad", float(loss_voxel_grad.detach().cpu()), global_step)

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

        z = torch.randn_like(z_gt, generator=g)
        current = z
        for t in ddim_scheduler.timesteps:
            t_tensor = torch.tensor([int(t)], device=device, dtype=torch.long)

            time_channel = dt_i.view(1, 1, 1, 1, 1).expand(1, 1, *z_start.shape[-3:])
            controlnet_cond = torch.cat([z_start, time_channel], dim=1)

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

        z_pred_unscaled = (current / scale_factor).clamp(-7, 7)
        z_gt_unscaled = (z_gt / scale_factor).clamp(-7, 7)
        z_start_unscaled = (z_start / scale_factor).clamp(-7, 7)

        pred_vol = _decode_latent_patchwise(
            ae, z_pred_unscaled, patch_size=int(decode_patch_size),
            overlap=float(decode_overlap), use_amp=bool(use_amp)
        )
        gt_vol = _decode_latent_patchwise(
            ae, z_gt_unscaled, patch_size=int(decode_patch_size),
            overlap=float(decode_overlap), use_amp=bool(use_amp)
        )
        st_vol = _decode_latent_patchwise(
            ae, z_start_unscaled, patch_size=int(decode_patch_size),
            overlap=float(decode_overlap), use_amp=bool(use_amp)
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
