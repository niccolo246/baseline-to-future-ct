import argparse
import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from datasets_three_d import LungLatentCSVDataset
from hybrid_vae_vitconv import ConvAEAdapter
from ct_generative_modelling.networks_shallow import init_latent_diffusion_ct

from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import DiffusionInferer

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from ct_generative_modelling.engine_latent_diffusion_ct import train_one_epoch, eval_one_epoch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
                self.shadow[name] = ((1.0 - self.decay) * param.data + self.decay * self.shadow[name]).clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
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
        shadow = state.get("shadow", None)
        if shadow is None:
            return

        if any(k.startswith("module.") for k in shadow.keys()):
            shadow = {k.replace("module.", "", 1): v for k, v in shadow.items()}

        # Use param dtype (not old shadow dtype)
        param_map = dict(self.model.named_parameters())

        missing = []
        for k in self.shadow.keys():
            if k in shadow:
                target_dtype = param_map[k].dtype if k in param_map else shadow[k].dtype
                self.shadow[k] = shadow[k].to(device=device, dtype=target_dtype).clone()
            else:
                missing.append(k)

        if missing:
            print(f"[EMA] WARNING: {len(missing)} keys missing when loading EMA state (showing 5): {missing[:5]}")

# ----------------------------------------------------------
# Arg parser
# ----------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser("Latent Diffusion CT training", add_help=False)

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="CSV list of latent .npy paths with latent-mask paths and covariates.",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="Validation CSV with latent-mask paths and covariates.",
    )

    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)

    # Model checkpoints
    parser.add_argument(
        "--ae_ckpt",
        type=str,
        required=True,
    )
    parser.add_argument("--diff_ckpt_init", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    # Optim
    parser.add_argument("--lr", type=float, default=3e-5, help="Max LR for cosine schedule.")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="Min LR for cosine schedule.")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs for LR schedule.")
    parser.add_argument("--accum_iter", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Logging / output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/latent_diffusion",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="outputs/latent_diffusion",
    )
    parser.add_argument("--save_every", type=int, default=1, help="Save every N epochs.")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    # Conditioning selector
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        choices=["semi", "full", "none"],
        default="semi",
        help="semi = allow NA but mask token + CFG dropout\n"
             "full = require complete covariates (drop NA rows)\n"
             "none = unconditional diffusion",
    )

    parser.add_argument(
        "--cond_dropout_prob",
        type=float,
        default=0.0,
        help="CFG dropout probability for 'semi' mode.",
    )

    # Optional override of covariate column names (otherwise dataset defaults)
    parser.add_argument(
        "--context_cols",
        nargs="+",
        default=None,
        help="(Optional) Context variable names in CSV. If None, uses LungLatentCSVDataset.DEFAULT_CONTEXT_COLS.",
    )

    # ----------------------------
    # Optional loss weighting
    # ----------------------------
    parser.add_argument(
        "--use_lung_weighting",
        type=misc.str2bool,
        default=False,
        help="Enable spatial lung weighting using batch['mask'] (latent grid).",
    )
    parser.add_argument(
        "--lung_weight",
        type=float,
        default=1.0,
        help="Weight multiplier inside lung region (outside stays 1.0).",
    )
    parser.add_argument(
        "--use_covariate_weighting",
        type=misc.str2bool,
        default=True,
        help="Up-weight samples where covariates are present (mask bit == 1 in context).",
    )
    parser.add_argument(
        "--covariate_present_weight",
        type=float,
        default=1.2,
        help="Sample weight multiplier when covariates are present.",
    )

    # ----------------------------
    # Scale factor control
    # ----------------------------
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.2913,
        help="If provided, uses this latent scale factor (e.g. 1/std from latent extraction). "
             "If not provided, it will be estimated from first --scale_factor_num_samples samples.",
    )
    parser.add_argument(
        "--scale_factor_num_samples",
        type=int,
        default=500,
        help="How many training samples to use for on-the-fly scale factor estimation (if --scale_factor not set).",
    )

    # ----------------------------
    # Visualization sampling
    # ----------------------------
    parser.add_argument(
        "--do_vis_sampling",
        type=misc.str2bool,
        default=True,
        help="If set, periodically sample latents during training, decode via AE, and save middle slices to log_dir.",
    )
    parser.add_argument("--vis_every", type=int, default=1)
    parser.add_argument("--vis_num_samples", type=int, default=2)
    parser.add_argument("--vis_num_inference_steps", type=int, default=300)
    parser.add_argument("--vis_seed", type=int, default=123)
    parser.add_argument("--vis_decode_patch_size", type=int, default=24)
    parser.add_argument("--vis_decode_overlap", type=float, default=0.25)
    parser.add_argument("--vis_use_amp", type=misc.str2bool, default=True)

    parser.add_argument(
    "--vis_use_ema",
    action="store_true",
    help="If set, visualization uses EMA weights; otherwise uses raw weights."
    )


    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    parser.add_argument("--clip_grad", type=float, default=1.0, help="Max gradient norm (e.g. 1.0 or 0.5) to prevent spikes.")

    parser.add_argument("--resume_ema", type=str, default=None)


    return parser



def _maybe_make_clean_csv(in_csv: str, out_csv: str, subset_cols, args) -> str:
    if misc.is_main_process():
        df = pd.read_csv(in_csv)
        df = df.dropna(subset=subset_cols)
        df.to_csv(out_csv, index=False)
        print(f"[data] Wrote cleaned CSV: {out_csv} (n={len(df)})")
    if args.distributed:
        torch.distributed.barrier()
    return out_csv


@torch.no_grad()
def _estimate_scale_factor(dataset_train, ae, device, num_samples: int) -> torch.Tensor:
    print(f"[scale_factor] Estimating from first {num_samples} training samples...")
    stds = []
    n = min(int(num_samples), len(dataset_train))
    if n <= 0:
        raise ValueError("Cannot estimate scale_factor: dataset_train is empty.")

    for i in range(n):
        sample = dataset_train[i]
        z = sample["latent"].to(device, dtype=torch.float32)
        if z.ndim == 5:
            z = z.squeeze(0)

        stds.append(torch.std(z))

    avg_std = torch.stack(stds).mean().clamp_min(1e-8)
    scale_factor = (1.0 / avg_std).to(device)
    print(f"[scale_factor] avg_std={avg_std.item():.6f} -> scale_factor={scale_factor.item():.6f}")
    return scale_factor


# ----------------------------------------------------------
# Safe resume hooks
# ----------------------------------------------------------
def _load_diffusion_checkpoint(
    *,
    resume_path: str,
    model_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: NativeScaler,
    device: torch.device,
    args,
    scale_factor: torch.Tensor,
):
    """
    Loads checkpoint on CPU (safe), then moves small tensors to GPU.
    Restores: model, optimizer, scaler, epoch, scale_factor (if args.scale_factor is None).
    Returns: (start_epoch, scale_factor)
    """
    if not resume_path or (not os.path.isfile(resume_path)):
        return 0, scale_factor

    print(f"[resume] Loading from: {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

    if "model" not in ckpt:
        raise KeyError(f"'model' key not found in checkpoint: {resume_path}")

    model_without_ddp.load_state_dict(ckpt["model"], strict=True)
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

    if args.scale_factor is None:
        sf = ckpt.get("scale_factor", None)
        if sf is not None:
            scale_factor = sf.to(device)
            print(f"[resume] Loaded scale_factor from ckpt: {scale_factor.item():.6f}")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"[resume] start_epoch = {start_epoch}")

    # cleanup
    del ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return start_epoch, scale_factor


# ----------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------
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
        if roi >= full: return [0]
        pos = list(range(0, full - roi + 1, stride))
        if pos[-1] != full - roi: pos.append(full - roi)
        return pos

    zs = _starts(D, patch_size)
    ys = _starts(H, patch_size)
    xs = _starts(W, patch_size)

    OD, OH, OW = D * 4, H * 4, W * 4
    out = torch.zeros((1, 1, OD, OH, OW), device="cpu", dtype=torch.float32)
    wgt = torch.zeros_like(out)

    # Handle DDP wrapper for AE if present
    inner_ae = ae.module if hasattr(ae, "module") else ae
    # Detect dtype
    model_dtype = next(inner_ae.parameters()).dtype

    # Probe
    zp_probe = z_unscaled[..., :patch_size, :patch_size, :patch_size].to(device, dtype=model_dtype).contiguous()
    with torch.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        # Access correct decode method
        if hasattr(inner_ae, 'ae'):
            yp_probe = inner_ae.ae.decode(zp_probe)
        else:
            yp_probe = inner_ae.decode(zp_probe)

        if isinstance(yp_probe, (tuple, list)):
            yp_probe = yp_probe[0]

    pD, pH, pW = yp_probe.shape[2:]
    w_patch = torch.ones((1, 1, pD, pH, pW), device="cpu", dtype=torch.float32)

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                # Force contiguous slice & match dtype
                zp = z_unscaled[..., z0:z0 + patch_size, y0:y0 + patch_size, x0:x0 + patch_size]
                zp = zp.to(device, dtype=model_dtype).contiguous()

                with torch.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    if hasattr(inner_ae, 'ae'):
                        yp = inner_ae.ae.decode(zp)
                    else:
                        yp = inner_ae.decode(zp)

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
def _sample_and_save_slices(
    *,
    ae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddim_scheduler: DDIMScheduler,
    scale_factor: torch.Tensor,
    fixed_context: torch.Tensor,
    out_dir: str,
    epoch: int,
    device: torch.device,
    num_samples: int = 2,
    num_inference_steps: int = 50,
    decode_patch_size: int = 24,
    decode_overlap: float = 0.25,
    use_amp: bool = True,
    seed: int = 123,
    log_writer: SummaryWriter = None,
    conditioning_mode: str = "semi",
    context_dim: int = 7,
):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    epoch_dir = os.path.join(out_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    # Set DDIM timesteps
    ddim_scheduler.set_timesteps(num_inference_steps=int(num_inference_steps))

    # Unwrap DDP if needed for config
    unwrapped_model = diffusion_model.module if hasattr(diffusion_model, "module") else diffusion_model
    latent_channels = getattr(unwrapped_model, "in_channels", None) or 4
    latent_shape = (latent_channels, 64, 64, 64)

    if conditioning_mode == "none":
        ctx = None
    else:
        if fixed_context is None:
            # Fallback to zeros if none provided
            ctx = torch.zeros((1, 1, int(context_dim)), device=device, dtype=torch.float32)
        else:
            ctx = fixed_context.to(device=device, dtype=torch.float32)
            # Normalize shape: (dim) -> (1, 1, dim) or (1, dim) -> (1, 1, dim)
            if ctx.ndim == 1:
                ctx = ctx.view(1, 1, -1)
            elif ctx.ndim == 2:
                ctx = ctx.unsqueeze(1)

            # Final safeguard for dim mismatch
            if ctx.shape[-1] != int(context_dim):
                print(f"[Vis Warning] Context dim mismatch: got {ctx.shape[-1]}, expected {context_dim}. Using zeros.")
                ctx = torch.zeros((1, 1, int(context_dim)), device=device, dtype=torch.float32)

    print(f"[Vis] Sampling {num_samples} images with {num_inference_steps} steps (DDIM)...")

    for i in range(num_samples):
        # 1. Noise
        z = torch.randn((1, *latent_shape), device=device, generator=g, dtype=torch.float32)

        # 2. Manual Sampling Loop
        current_img = z
        for t in ddim_scheduler.timesteps:
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)

            with torch.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                noise_pred = diffusion_model(current_img, timesteps=t_tensor, context=ctx)

            current_img, _ = ddim_scheduler.step(noise_pred, t, current_img)
            current_img = current_img.clamp(
                                            -7.0 * float(scale_factor.item()),
                                             7.0 * float(scale_factor.item())
                                            )

        # 3. Unscale
        z_unscaled = current_img / scale_factor.to(device=device, dtype=torch.float32)
        if z_unscaled.ndim == 4:
            z_unscaled = z_unscaled.unsqueeze(0)


        # Clamp to the training latent range before AE decode.
        z_min, z_max = z_unscaled.min().item(), z_unscaled.max().item()
        print(f"[Vis] latent pre-clamp: min={z_min:.3f} max={z_max:.3f}")
        z_unscaled = z_unscaled.clamp(-7.0, 7.0)
        z_min2, z_max2 = z_unscaled.min().item(), z_unscaled.max().item()
        print(f"[Vis] latent post-clamp: min={z_min2:.3f} max={z_max2:.3f}")


        # 4. Decode
        recon_logits = _decode_latent_patchwise(
            ae=ae,
            z_unscaled=z_unscaled,
            patch_size=int(decode_patch_size),
            overlap=float(decode_overlap),
            use_amp=bool(use_amp),
        )

        recon_logits = recon_logits.clamp(-1.0, 1.0)
        recon_01 = ((recon_logits + 1.0) * 0.5).clamp(0.0, 1.0)
        vol = recon_01[0, 0].numpy()

        D, H, W = vol.shape
        axial = vol[D // 2, :, :]
        coronal = vol[:, H // 2, :]
        sagittal = vol[:, :, W // 2]

        axial_u8 = _to_uint8(axial)
        cor_u8 = _to_uint8(coronal)
        sag_u8 = _to_uint8(sagittal)

        for name, arr_u8 in [("axial", axial_u8), ("coronal", cor_u8), ("sagittal", sag_u8)]:
            out_png = os.path.join(epoch_dir, f"sample_{i:02d}_{name}.png")
            plt.imsave(out_png, arr_u8, cmap="gray", vmin=0, vmax=255)

        if log_writer is not None:
            ax_t = torch.from_numpy(axial_u8).unsqueeze(0)
            co_t = torch.from_numpy(cor_u8).unsqueeze(0)
            sa_t = torch.from_numpy(sag_u8).unsqueeze(0)
            log_writer.add_image(f"vis_samples/epoch_{epoch:04d}/sample_{i:02d}_axial", ax_t, epoch, dataformats="CHW")
            log_writer.add_image(f"vis_samples/epoch_{epoch:04d}/sample_{i:02d}_coronal", co_t, epoch, dataformats="CHW")
            log_writer.add_image(f"vis_samples/epoch_{epoch:04d}/sample_{i:02d}_sagittal", sa_t, epoch, dataformats="CHW")

    if log_writer is not None:
        log_writer.flush()


def _get_fixed_context_for_vis(dataset_val, dataset_train, device: torch.device):
    ds = dataset_val if dataset_val is not None else dataset_train
    if ds is None or len(ds) == 0:
        return None
    try:
        s = ds[0]
        ctx = s.get("context", None)
        if ctx is None:
            return None
        return ctx.to(device=device, dtype=torch.float32)
    except Exception:
        return None


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main(args):
    misc.init_distributed_mode(args)
    print("Args:\n", "{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Context dim
    c_cols = args.context_cols if args.context_cols is not None else LungLatentCSVDataset.DEFAULT_CONTEXT_COLS
    current_context_dim = len(c_cols) + 1
    print(f"[Context] cols={len(c_cols)} + mask_token=1 => context_dim={current_context_dim}")

    print("[data] Using latent .npy (CSV-backed)")
    train_csv = args.data_path
    val_csv = args.val_data_path

    if args.conditioning_mode == "full":
        clean_train = os.path.join(args.output_dir, "clean_train.csv")
        train_csv = _maybe_make_clean_csv(args.data_path, clean_train, c_cols, args)
        if val_csv is not None:
            clean_val = os.path.join(args.output_dir, "clean_val.csv")
            val_csv = _maybe_make_clean_csv(args.val_data_path, clean_val, c_cols, args)

    dataset_train = LungLatentCSVDataset(
        train_csv,
        context_cols=args.context_cols,
        conditioning_mode=args.conditioning_mode,
    )

    dataset_val = None
    if val_csv is not None:
        dataset_val = LungLatentCSVDataset(
            val_csv,
            context_cols=args.context_cols,
            conditioning_mode=args.conditioning_mode,
        )

    # Samplers / loaders
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = (
            torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
            if dataset_val is not None else None
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = None
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # Load AE (frozen)
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


    if args.ae_ckpt is not None and os.path.isfile(args.ae_ckpt):
        print(f"[AE] Loading checkpoint from: {args.ae_ckpt}")
        try:
            ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
            if isinstance(ae_ckpt, dict) and "model_without_ddp" in ae_ckpt:
                ae.load_state_dict(ae_ckpt["model_without_ddp"], strict=False)
            elif isinstance(ae_ckpt, dict) and "model" in ae_ckpt:
                ae.load_state_dict(ae_ckpt["model"], strict=False)
            else:
                ae.load_state_dict(ae_ckpt, strict=False)
            print("[AE] Loaded successfully (float32).")
        except Exception as e:
            print(f"[AE] Error loading checkpoint: {e}")
            print("[AE] Continuing with RANDOM weights (Visualization will be noise).")
    else:
        print(f"[AE] No checkpoint provided (args.ae_ckpt={args.ae_ckpt}).")
        print("[AE] Model has RANDOM weights. Visualization images will be noise.")

    # Force float32 to prevent half-precision errors in decode
    ae.float()
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    print("[AE] Loaded and frozen (float32)")

    # Diffusion model
    diffusion = init_latent_diffusion_ct(
        args.diff_ckpt_init,
        args=args,
        context_dim=current_context_dim,
    ).to(device)

    if args.distributed:
        diffusion = torch.nn.parallel.DistributedDataParallel(
            diffusion, device_ids=[args.gpu], find_unused_parameters=False
        )
        diffusion_without_ddp = diffusion.module
    else:
        diffusion_without_ddp = diffusion

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0205,
        clip_sample=False,
        prediction_type="epsilon",
    )

    inferer = DiffusionInferer(scheduler=scheduler)

    # Optim
    optimizer = torch.optim.AdamW(
        diffusion_without_ddp.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    loss_scaler = NativeScaler()

    # Scale factor
    if args.scale_factor is not None:
        scale_factor = torch.tensor(float(args.scale_factor), device=device)
        if misc.is_main_process():
            print(f"[scale_factor] Using provided --scale_factor = {scale_factor.item():.6f}")
    else:
        scale_factor = torch.tensor(1.0, device=device)

    # Logging
    log_writer = None
    if (not args.distributed or misc.is_main_process()) and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # Resume (safe)
    start_epoch = 0
    if args.resume is not None:
        start_epoch, scale_factor = _load_diffusion_checkpoint(
            resume_path=args.resume,
            model_without_ddp=diffusion_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            device=device,
            args=args,
            scale_factor=scale_factor,
        )
    # ----------------------------------------------------------
    # Initialize EMA (after resume)
    # ----------------------------------------------------------
    ema = EMA(diffusion_without_ddp, decay=0.9999)
    print("[EMA] Initialized Exponential Moving Average (0.9999) from CURRENT weights")

    loaded_ema = False

    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"], device=device)
            print("[resume] Loaded EMA running state (shadow + decay).")
            loaded_ema = True
        del ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Only use --resume_ema if we DID NOT already restore EMA running state
    if (not loaded_ema) and (args.resume_ema is not None) and os.path.isfile(args.resume_ema):
        print(f"[resume_ema] Loading EMA weights from: {args.resume_ema}")
        ckpt_ema = torch.load(args.resume_ema, map_location="cpu", weights_only=False)
        print(f"[resume_ema] Loaded EMA ckpt epoch={ckpt_ema.get('epoch', 'NA')}")

        ema_sd = ckpt_ema.get("model", None)
        if ema_sd is None:
            raise KeyError(f"[resume_ema] 'model' key not found in {args.resume_ema}")

        if any(k.startswith("module.") for k in ema_sd.keys()):
            ema_sd = {k.replace("module.", "", 1): v for k, v in ema_sd.items()}

        param_map = dict(diffusion_without_ddp.named_parameters())

        missing = []
        for name in ema.shadow.keys():
            if name in ema_sd:
                ema.shadow[name] = ema_sd[name].to(device=device, dtype=param_map[name].dtype).clone()
            else:
                missing.append(name)

        if missing:
            print(f"[resume_ema] WARNING: {len(missing)} EMA params missing in checkpoint (showing up to 5): {missing[:5]}")

        if "ema_decay" in ckpt_ema:
            ema.decay = float(ckpt_ema["ema_decay"])

        print("[resume_ema] Seeded EMA shadow from EMA checkpoint.")

        del ckpt_ema
        gc.collect()

    # Scale factor estimate (if not provided and not loaded)
    if args.scale_factor is None and (scale_factor is None or float(scale_factor.item()) == 1.0):
        if misc.is_main_process():
            scale_factor = _estimate_scale_factor(
                dataset_train=dataset_train,
                ae=ae,
                device=device,
                num_samples=args.scale_factor_num_samples,
            )
        else:
            scale_factor = torch.tensor(1.0, device=device)

    if args.distributed:
        torch.distributed.broadcast(scale_factor, src=0)

    # Fixed context for vis
    fixed_context = None
    if args.do_vis_sampling:
        fixed_context = _get_fixed_context_for_vis(dataset_val, dataset_train, device)

    vis_ddim = DDIMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0205,
        clip_sample=False,
        prediction_type="epsilon",
    )

    # Safety check: latents should be clamped to the training range.
    if misc.is_main_process():
        print("\n[VERIFICATION] Checking first batch for clamping...")
        try:
            check_batch = next(iter(data_loader_train))

            # Auto-detect the key
            if "latent" in check_batch:
                tensor_to_check = check_batch["latent"]
                key_name = "latent"
            elif "starting_latent" in check_batch:
                tensor_to_check = check_batch["starting_latent"]
                key_name = "starting_latent"
            else:
                raise KeyError(f"Could not find 'latent' or 'starting_latent'. Keys found: {list(check_batch.keys())}")

            v_min, v_max = tensor_to_check.min().item(), tensor_to_check.max().item()
            print(f"Checking '{key_name}' - Min: {v_min:.4f}, Max: {v_max:.4f}")

            if v_max > 7.1 or v_min < -7.1:
                raise ValueError(f"Data is not clamped to [-7, 7]. Found range [{v_min}, {v_max}]")
            else:
                print("[VERIFICATION] PASSED. Data is safely clamped to [-7, 7].\n")

        except Exception as e:
            print(f"[VERIFICATION] FAILED: {e}\n")
            import sys
            sys.exit(1)


    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            if data_loader_val is not None and isinstance(
                data_loader_val.sampler, torch.utils.data.DistributedSampler
            ):
                data_loader_val.sampler.set_epoch(epoch)

        # Pass EMA to train_one_epoch
        train_stats = train_one_epoch(
            ae=ae,
            diffusion=diffusion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            scheduler=scheduler,
            inferer=inferer,
            scale_factor=scale_factor,
            log_writer=log_writer,
            use_latent_npz=True,
            cond_dropout_prob=args.cond_dropout_prob,
            args=args,
            ema=ema  # <--- PASS EMA HERE
        )

        val_stats = {}
        if data_loader_val is not None:
            val_stats = eval_one_epoch(
                ae=ae,
                diffusion=diffusion,
                data_loader=data_loader_val,
                device=device,
                epoch=epoch,
                scheduler=scheduler,
                inferer=inferer,
                scale_factor=scale_factor,
                log_writer=log_writer,
                use_latent_npz=True,
                args=args,
            )

        # Vis sampling (Using EMA weights)
        # Vis sampling (RAW or EMA depending on flag)
        if args.do_vis_sampling and misc.is_main_process():
            if (epoch % max(1, int(args.vis_every))) == 0:
                vis_root = os.path.join(args.log_dir if args.log_dir else args.output_dir, "vis_samples")
                use_ema_for_vis = bool(getattr(args, "vis_use_ema", False))

                if use_ema_for_vis:
                    ema.apply_shadow()
                    print(f"[Vis] Swapped to EMA weights for sampling at epoch {epoch}")
                    vis_root = os.path.join(vis_root, "ema")
                else:
                    print(f"[Vis] Using RAW weights for sampling at epoch {epoch}")
                    vis_root = os.path.join(vis_root, "raw")

                try:
                    diffusion_without_ddp.eval()
                    _sample_and_save_slices(
                        ae=ae,
                        diffusion_model=diffusion_without_ddp,
                        ddim_scheduler=vis_ddim,
                        scale_factor=scale_factor,
                        fixed_context=fixed_context,
                        out_dir=vis_root,
                        epoch=epoch,
                        device=device,
                        num_samples=int(args.vis_num_samples),
                        num_inference_steps=int(args.vis_num_inference_steps),
                        decode_patch_size=int(args.vis_decode_patch_size),
                        decode_overlap=float(args.vis_decode_overlap),
                        use_amp=bool(args.vis_use_amp),
                        seed=int(args.vis_seed),
                        log_writer=log_writer,
                        conditioning_mode=str(args.conditioning_mode),
                        context_dim=int(current_context_dim),
                    )
                except Exception as e:
                    print(f"[vis_sampling][WARN] Failed at epoch {epoch}: {repr(e)}")
                    import traceback
                    traceback.print_exc()
                finally:
                    diffusion_without_ddp.train()
                    if use_ema_for_vis:
                        ema.restore()
                        print("[Vis] Restored RAW weights after EMA sampling")


        # Save checkpoint
        if args.output_dir and misc.is_main_process():
            # 1. Standard Checkpoint (Every Epoch) - SAVES RAW WEIGHTS (Safe for Resume)
            if ((epoch + 1) % int(args.save_every) == 0) or ((epoch + 1) == args.epochs):
                ckpt_path = os.path.join(args.output_dir, f"latent_diffusion_epoch{epoch}.pth")
                to_save = {
                    "epoch": epoch,
                    "model": diffusion_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                    "scale_factor": scale_factor.detach().cpu(),
                    #"ema": ema.state_dict(),
                }
                torch.save(to_save, ckpt_path)
                print(f"[epoch {epoch}] Saved checkpoint -> {ckpt_path}")

            # 2. EMA Checkpoint (Every 5 Epochs) - SAVES SMOOTH WEIGHTS (Strategy: Swap -> Save -> Restore)
            if (epoch + 1) % 5 == 0:
                ema_path = os.path.join(args.output_dir, f"diffusion_ema_epoch{epoch}.pth")

                # A. Swap to EMA weights
                ema.apply_shadow()

                # B. Create standard state dict (now contains EMA weights!)
                to_save_ema = {
                    "epoch": epoch,
                    "model": diffusion_without_ddp.state_dict(), # This is now the smooth model
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                    "scale_factor": scale_factor.detach().cpu(),
                    "is_ema": True, # Optional flag for reference
                    "ema": ema.state_dict(),
                }

                # C. Save
                torch.save(to_save_ema, ema_path)
                print(f"[epoch {epoch}] Saved EMA weights (standard format) -> {ema_path}")

                # D. Restore Raw weights (Critical for training to continue)
                ema.restore()

            # Write log line
            log_stats = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "scale_factor": float(scale_factor.detach().cpu().item()),
            }
            with open(os.path.join(args.output_dir, "log_latent_diffusion.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    print(f"Training time: {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Latent Diffusion CT training",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)
