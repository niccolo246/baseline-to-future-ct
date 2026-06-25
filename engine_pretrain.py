# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Patch-based HybridVAE training engine:
# multi-term loss + KL warmup + denoising + optional GAN
# Autoencoder training engine for [-1,1] CT patches.

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

import util.lr_sched as lr_sched
from util.misc import MetricLogger, SmoothedValue


# -------------------------
# Range helpers ([-1,1])
# -------------------------
def _sanitize_clamp_pm1(x: torch.Tensor) -> torch.Tensor:
    """NaN/Inf guard + clamp to [-1,1]."""
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return x.clamp(-1.0, 1.0)


def _pm1_to_01(x: torch.Tensor) -> torch.Tensor:
    """Map [-1,1] -> [0,1] (useful if perceptual net expects [0,1])."""
    return (x + 1.0) * 0.5


# -------------------------
# Loss: lung-weighted L1 + optional lung-window emphasis
# -------------------------
def lung_weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    lung_mask: torch.Tensor,
    lung_w: float = 2.0,
    nonlung_w: float = 1.0,
    use_lung_window: bool = True,
    lung_window_low: float = -0.8,
    lung_window_high: float = 0.5,
    lung_window_weight: float = 2.0,
    lung_outside_window_weight: float = 1.0,
):
    if lung_mask is None:
        # Keep the loss reduction in fp32 for stability.
        return F.l1_loss(pred.float(), target.float())

    # force fp32 for stable reductions
    pred_f   = pred.float()
    target_f = target.float()
    mask_f   = lung_mask.to(dtype=torch.float32)

    diff = (pred_f - target_f).abs()

    w = mask_f * lung_w + (1.0 - mask_f) * nonlung_w

    if use_lung_window:
        in_window = ((target_f >= lung_window_low) & (target_f <= lung_window_high)).to(torch.float32)
        lung_factor = in_window * lung_window_weight + (1.0 - in_window) * lung_outside_window_weight
        w = w * (mask_f * lung_factor + (1.0 - mask_f))

    # fp32 sums (no fp16 overflow)
    return (diff * w).sum() / (w.sum() + 1e-6)



# -------------------------
# Schedules / metrics
# -------------------------
def _beta_schedule(step: int, warmup: int, final: float = 1.0) -> float:
    if warmup <= 0:
        return final
    if step >= warmup:
        return final
    return final * (step / warmup)


def _grad3d_l1_diff(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """|∇y - ∇x| averaged over D/H/W (finite differences)."""
    def diffs(t):
        dz = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]
        dy = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]
        dx = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]
        return dz, dy, dx

    yz, yy, yx = diffs(y)
    xz, xy, xx = diffs(x)
    return ((yz - xz).abs().mean() + (yy - xy).abs().mean() + (yx - xx).abs().mean()) / 3.0


_SSIM_CACHE = {}
def _gauss_kernel_3d_cached(size: int, sigma: float, device, dtype):
    key = (size, float(sigma), device.type, device.index, str(dtype))
    k = _SSIM_CACHE.get(key)
    if k is None:
        coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
        g1 = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g1 = g1 / g1.sum()
        g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
        k = g3[None, None]  # (1,1,S,S,S)
        _SSIM_CACHE[key] = k
    return k


def _ssim3d(x: torch.Tensor, y: torch.Tensor, size: int = 7, sigma: float = 1.5, data_range: float = 2.0):
    """
    SSIM for 3D volumes.
    For data in [-1,1], data_range MUST be 2.0.
    """
    k = _gauss_kernel_3d_cached(size, sigma, x.device, x.dtype)
    pad = size // 2
    mu_x = F.conv3d(x, k, padding=pad)
    mu_y = F.conv3d(y, k, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv3d(x * x, k, padding=pad) - mu_x2
    sigma_y2 = F.conv3d(y * y, k, padding=pad) - mu_y2
    sigma_xy = F.conv3d(x * y, k, padding=pad) - mu_xy

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / (den + 1e-12)).mean()


def _kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    args=None,
    perceptual_model=None,
):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss',   SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('recon',  SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('kl',     SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('l1',     SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('ssim',   SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('grad',   SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('percep', SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('gan_g',  SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('gan_d',  SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('beta_t', SmoothedValue(window_size=10, fmt='{value:.4f}'))

    header = f'Val Epoch: [{epoch}]'

    l1_w       = getattr(args, 'l1_w', 1.0)
    ssim_w     = getattr(args, 'ssim_w', 0.0)
    grad_w     = getattr(args, 'grad_w', 0.0)
    kl_global_w = getattr(args, "kl_weight", 0.1)
    percep_w   = getattr(args, "perceptual_weight", 0.0)

    use_ssim = ssim_w > 0.0
    use_grad = grad_w > 0.0

    # lung window params (in [-1,1])
    use_lung_window = getattr(args, "use_lung_window", True)
    lung_window_low = getattr(args, "lung_window_low", -0.8)
    lung_window_high = getattr(args, "lung_window_high", 0.5)
    lung_window_weight = getattr(args, "lung_window_weight", 2.0)
    lung_outside_window_weight = getattr(args, "lung_outside_window_weight", 1.0)

    # Eval: beta_t=1 unless disable_kl
    beta_t = 0.0 if getattr(args, "disable_kl", False) else 1.0

    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
    amp_dtype  = torch.float16 if amp_device == 'cuda' else torch.bfloat16

    for _, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, lung_mask = batch
        else:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            lung_mask = None

        x = _sanitize_clamp_pm1(x.to(device, non_blocking=True))
        if lung_mask is not None:
            lung_mask = (lung_mask > 0.5).to(device, non_blocking=True).float()
            #lung_mask = lung_mask.to(device, non_blocking=True)

        with torch.autocast(device_type=amp_device, dtype=amp_dtype):
            # ConvAEAdapter returns (y, y_ds2, mu, logvar).
            y, _, mu, logvar = model(
                x,
                mask_ratio=getattr(args, 'mask_ratio', 0.0),
                use_mean=(beta_t == 0.0),
            )

        y = _sanitize_clamp_pm1(y)

        l1 = lung_weighted_l1(
            y, x, lung_mask,
            lung_w=getattr(args, 'lung_l1_weight', 2.0),
            nonlung_w=getattr(args, 'nonlung_l1_weight', 1.0),
            use_lung_window=use_lung_window,
            lung_window_low=lung_window_low,
            lung_window_high=lung_window_high,
            lung_window_weight=lung_window_weight,
            lung_outside_window_weight=lung_outside_window_weight,
        )

        perceptual_loss = torch.tensor(0.0, device=device)
        if perceptual_model is not None and percep_w > 0.0:
            x01 = _pm1_to_01(x)
            y01 = _pm1_to_01(y)
            with torch.no_grad():
                feat_real = perceptual_model.extract_perceptual(x01.float())
            feat_recon = perceptual_model.extract_perceptual(y01.float())
            perceptual_loss = F.mse_loss(feat_recon, feat_real)

        with torch.cuda.amp.autocast(enabled=False):
            ssim = torch.tensor(0.0, device=device)
            grad = torch.tensor(0.0, device=device)

            if use_ssim:
                ssim = _ssim3d(y.float(), x.float(), data_range=2.0)
            if use_grad:
                grad = _grad3d_l1_diff(y.float(), x.float())

            recon = (
                l1_w * l1.float()
                + (ssim_w * (1.0 - ssim) if use_ssim else 0.0)
                + (grad_w * grad if use_grad else 0.0)
                + percep_w * perceptual_loss.float()
            )

        kl_val = torch.tensor(0.0, device=device)
        if beta_t > 0.0:
            kl_val = _kld(mu.float(), logvar.float())

        loss = recon + beta_t * kl_val * kl_global_w

        metric_logger.update(
            loss=float(loss.detach()),
            recon=float(recon.detach()),
            kl=float(kl_val.detach()),
            l1=float(l1.detach()),
            ssim=float(ssim.detach()),
            grad=float(grad.detach()),
            percep=float(perceptual_loss.detach()),
            gan_g=0.0,
            gan_d=0.0,
            beta_t=float(beta_t),
        )

    metric_logger.synchronize_between_processes()
    print("Val averaged stats:", {k: m.global_avg for k, m in metric_logger.meters.items()})
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# -------------------------
# Train
# -------------------------
def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    ema=None,
    perceptual_model=None,
    discriminator=None,
    disc_optimizer=None,
):
    model.train()
    if discriminator is not None:
        discriminator.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss',   SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('recon',  SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('kl',     SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('l1',     SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('ssim',   SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('grad',   SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('percep', SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('gan_g',  SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('gan_d',  SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('lr',     SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('beta_t', SmoothedValue(window_size=10, fmt='{value:.4f}'))

    header = f'Epoch: [{epoch}]'

    accum_iter    = getattr(args, 'accum_iter', 1)
    denoise_sigma = getattr(args, 'denoise_sigma', 0.0)
    mask_ratio    = getattr(args, 'mask_ratio', 0.0)
    beta_warm     = getattr(args, 'beta_warmup', 40000)

    l1_w          = getattr(args, 'l1_w', 1.0)
    ssim_w        = getattr(args, 'ssim_w', 0.0)
    grad_w        = getattr(args, 'grad_w', 0.0)

    percep_base_w = getattr(args, "perceptual_weight", 0.0)
    gan_base_w    = getattr(args, "gan_weight", 0.0)
    percep_start  = getattr(args, "percep_start_epoch", 0)
    percep_ramp   = max(1, getattr(args, "percep_ramp_epochs", 1))
    gan_start     = getattr(args, "gan_start_epoch", 0)
    gan_ramp      = max(1, getattr(args, "gan_ramp_epochs", 1))

    kl_global_w   = getattr(args, "kl_weight", 0.1)

    use_ssim = ssim_w > 0.0
    use_grad = grad_w > 0.0

    # lung window params (in [-1,1])
    use_lung_window = getattr(args, "use_lung_window", True)
    lung_window_low = getattr(args, "lung_window_low", -0.8)
    lung_window_high = getattr(args, "lung_window_high", 0.5)
    lung_window_weight = getattr(args, "lung_window_weight", 2.0)
    lung_outside_window_weight = getattr(args, "lung_outside_window_weight", 1.0)

    # ramps
    percep_factor = 0.0
    if epoch >= percep_start:
        percep_factor = min(1.0, (epoch - percep_start) / percep_ramp)
    effective_percep_w = percep_base_w * percep_factor

    gan_factor = 0.0
    if epoch >= gan_start:
        gan_factor = min(1.0, (epoch - gan_start) / gan_ramp)
    effective_gan_w = gan_base_w * gan_factor

    use_gan = (
        discriminator is not None
        and disc_optimizer is not None
        and effective_gan_w > 0.0
    )

    optimizer.zero_grad(set_to_none=True)

    start_step  = epoch * len(data_loader)
    global_step = start_step

    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, lung_mask = batch
        else:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            lung_mask = None

        x = x.to(device, non_blocking=True)
        x = _sanitize_clamp_pm1(x)
        if lung_mask is not None:
            lung_mask = (lung_mask > 0.5).to(device, non_blocking=True).float()
            #lung_mask = lung_mask.to(device, non_blocking=True)

        # LR schedule
        if (step % accum_iter) == 0:
            cur_epoch = epoch + step / len(data_loader)
            cur_lr = lr_sched.adjust_learning_rate(optimizer, cur_epoch, args)
        else:
            cur_lr = optimizer.param_groups[0]["lr"]

        # Denoising
        if denoise_sigma > 0:
            noisy_x = (x + torch.randn_like(x) * denoise_sigma).clamp(-1.0, 1.0)
        else:
            noisy_x = x

        amp_device = 'cuda' if x.device.type == 'cuda' else 'cpu'
        amp_dtype  = torch.float16 if amp_device == 'cuda' else torch.bfloat16

        # KL warmup
        if getattr(args, "disable_kl", False):
            beta_t = 0.0
        else:
            beta_t = _beta_schedule(global_step, beta_warm, 1.0)

        use_mean = (getattr(args, "disable_kl", False) or beta_t == 0.0)

        # Forward
        with torch.autocast(device_type=amp_device, dtype=amp_dtype):
            y, _, mu, logvar = model(noisy_x, mask_ratio=mask_ratio, use_mean=use_mean)

        y = _sanitize_clamp_pm1(y)

        # L1
        l1 = lung_weighted_l1(
            y, x, lung_mask,
            lung_w=getattr(args, 'lung_l1_weight', 2.0),
            nonlung_w=getattr(args, 'nonlung_l1_weight', 1.0),
            use_lung_window=use_lung_window,
            lung_window_low=lung_window_low,
            lung_window_high=lung_window_high,
            lung_window_weight=lung_window_weight,
            lung_outside_window_weight=lung_outside_window_weight,
        )

        # Perceptual (optional)
        perceptual_loss = torch.tensor(0.0, device=device)
        if (
            perceptual_model is not None
            and effective_percep_w > 0.0
            and (step % getattr(args, "perceptual_every", 1) == 0)
        ):
            x01 = _pm1_to_01(x)
            y01 = _pm1_to_01(y)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                feat_real = perceptual_model.extract_perceptual(x01.float())
            with torch.cuda.amp.autocast(enabled=False):
                feat_recon = perceptual_model.extract_perceptual(y01.float())
            perceptual_loss = F.mse_loss(feat_recon, feat_real)

        # GAN (optional)
        gan_loss_G = torch.tensor(0.0, device=device)
        gan_loss_D = torch.tensor(0.0, device=device)

        if use_gan:
            disc_optimizer.zero_grad(set_to_none=True)

            real_input = x.detach().float()
            fake_input_d = y.detach().float()

            real_logits = discriminator(real_input)
            fake_logits = discriminator(fake_input_d)

            d_loss_real = F.relu(1.0 - real_logits).mean()
            d_loss_fake = F.relu(1.0 + fake_logits).mean()
            gan_loss_D = d_loss_real + d_loss_fake

            gan_loss_D.backward()
            disc_optimizer.step()

            fake_logits_g = discriminator(y.float())
            gan_loss_G = -fake_logits_g.mean()

        # SSIM / grad in fp32
        with torch.cuda.amp.autocast(enabled=False):
            ssim = torch.tensor(0.0, device=device)
            grad = torch.tensor(0.0, device=device)

            if use_ssim:
                ssim = _ssim3d(y.float(), x.float(), data_range=2.0)
            if use_grad:
                grad = _grad3d_l1_diff(y.float(), x.float())

            recon = (
                l1_w * l1.float()
                + (ssim_w * (1.0 - ssim) if use_ssim else 0.0)
                + (grad_w * grad if use_grad else 0.0)
                + effective_percep_w * perceptual_loss.float()
                + (effective_gan_w * gan_loss_G.float() if use_gan else 0.0)
            )

        # KL
        kl_val = torch.tensor(0.0, device=device)
        if beta_t > 0.0:
            kl_val = _kld(mu.float(), logvar.float())

        loss = recon + beta_t * kl_val * kl_global_w

        # Non-finite guard
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            print(f"[Step {step}] non-finite loss={loss_value}; skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backprop (scaled, accumulation)
        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=((step + 1) % accum_iter == 0),
        )

        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model.module if hasattr(model, "module") else model)

        # Logging
        metric_logger.update(
            loss=loss_value,
            recon=float(recon.detach()),
            kl=float(kl_val.detach()),
            l1=float(l1.detach()),
            ssim=float(ssim.detach()),
            grad=float(grad.detach()),
            percep=float(perceptual_loss.detach()),
            gan_g=float(gan_loss_G.detach()),
            gan_d=float(gan_loss_D.detach()),
            lr=cur_lr,
            beta_t=float(beta_t),
        )

        if log_writer is not None and (step % 20 == 0 or step == len(data_loader) - 1):
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train/recon', float(recon.detach()), epoch_1000x)
            log_writer.add_scalar('train/kl', float(kl_val.detach()), epoch_1000x)
            log_writer.add_scalar('train/l1', float(l1.detach()), epoch_1000x)
            log_writer.add_scalar('train/ssim', float(ssim.detach()), epoch_1000x)
            log_writer.add_scalar('train/grad', float(grad.detach()), epoch_1000x)
            log_writer.add_scalar('train/percep', float(perceptual_loss.detach()), epoch_1000x)
            log_writer.add_scalar('train/gan_g', float(gan_loss_G.detach()), epoch_1000x)
            log_writer.add_scalar('train/gan_d', float(gan_loss_D.detach()), epoch_1000x)
            log_writer.add_scalar('train/lr', cur_lr, epoch_1000x)
            log_writer.add_scalar('train/beta_t', float(beta_t), epoch_1000x)

        global_step += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", {k: m.global_avg for k, m in metric_logger.meters.items()})
    return {k: m.global_avg for k, m in metric_logger.meters.items()}
