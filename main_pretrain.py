# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from datasets_three_d import Custom3DDDatasetMask
from monai.transforms import RandCropByPosNegLabeld, RandSpatialCropd

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain import train_one_epoch, eval_one_epoch

from hybrid_vae_vitconv import ConvAEAdapter, PatchDiscriminator3D

try:
    import bitsandbytes as bnb
    OptimCls = bnb.optim.AdamW8bit
except ImportError:
    from torch.optim import AdamW
    OptimCls = AdamW


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                sd[k].copy_(v)



def get_args_parser():
    parser = argparse.ArgumentParser("Patch autoencoder training", add_help=False)

    parser.add_argument('--batch_size', default=6, type=int,
                        help='Batch size per GPU (effective = batch_size * accum_iter * #gpus)')

    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    parser.add_argument('--input_size', default=96, type=int,
                        help='Input patch size (cubic); paper uses 96^3.')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Text file with preprocessed CT .npz paths.')

    # Optimizer / schedule
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Global ref LR used by the scheduler.')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='Base LR')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=2)

    # Output / logging / resume
    parser.add_argument('--output_dir', default='outputs/autoencoder',
                        help='Where to save checkpoints')
    parser.add_argument('--log_dir', default='outputs/autoencoder',
                        help='Where to write TensorBoard logs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None)

    parser.add_argument('--perceptual_ckpt', type=str, default=None)
    parser.add_argument('--perceptual_weight', type=float, default=0.0,
                        help='Weight of the perceptual loss term (0 = disabled).')
    parser.add_argument('--perceptual_every', type=int, default=5)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # VAE pretext controls
    parser.add_argument('--mask_ratio', default=0.0, type=float)
    parser.add_argument('--denoise_sigma', default=0.0, type=float)

    parser.add_argument('--beta_warmup', default=14000, type=int)

    parser.add_argument('--kl_weight', default=1e-6, type=float)

    parser.add_argument('--l1_w', default=1.0, type=float)

    parser.add_argument('--ssim_w', default=0.02, type=float)
    parser.add_argument('--grad_w', default=0.02, type=float)
    parser.add_argument('--disable_kl', type=misc.str2bool, default=False)


    parser.add_argument('--use_ema', type=misc.str2bool, default=False)
    parser.add_argument('--ema_decay', default=0.999, type=float)

    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--optim8bit', type=misc.str2bool, default=True)
    parser.add_argument('--head_lr', type=float, default=1e-4)

    # PatchGAN term used in the paper autoencoder objective.
    parser.add_argument('--use_gan', type=misc.str2bool, default=True)
    parser.add_argument('--gan_weight', type=float, default=0.002)
    parser.add_argument('--gan_lr', type=float, default=1e-4)

    parser.add_argument('--percep_start_epoch', type=int, default=5)
    parser.add_argument('--percep_ramp_epochs', type=int, default=5)
    parser.add_argument('--gan_start_epoch', type=int, default=60)
    parser.add_argument('--gan_ramp_epochs', type=int, default=10)


    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=1)

    parser.add_argument('--use_lung_mask', type=misc.str2bool, default=True)
    parser.add_argument('--lung_l1_weight', type=float, default=2.0)
    parser.add_argument('--nonlung_l1_weight', type=float, default=1.0)

    parser.add_argument('--summit_only', type=misc.str2bool, default=False)

    # Windowing args for lung focus
    parser.add_argument('--use_lung_window', type=misc.str2bool, default=True)
    parser.add_argument('--lung_window_low', type=float, default=-0.8)
    parser.add_argument('--lung_window_high', type=float, default=0.5)
    parser.add_argument('--lung_window_weight', type=float, default=2.0)
    parser.add_argument('--lung_outside_window_weight', type=float, default=1.0)


    return parser


class PatchCropTransform:
    """
    Expects input as:
      (img, mask) where img: [1,D,H,W], mask: [1,D,H,W] (float 0/1)
    Returns:
      (img_crop, mask_crop) cropped to patch_size^3.
    """

    def __init__(self, patch_size, pos=4, neg=1, num_samples=1):
        self.patch_size = (patch_size, patch_size, patch_size)

        self.smart = RandCropByPosNegLabeld(
            keys=("img", "mask"),
            label_key="mask",
            spatial_size=self.patch_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_key="img",
            image_threshold=0,
        )

        self.fallback = RandSpatialCropd(
            keys=("img", "mask"),
            roi_size=self.patch_size,
            random_size=False,
        )

    def __call__(self, sample):
        # sample can be img alone or (img, mask)
        if isinstance(sample, (tuple, list)):
            img, mask = sample
        else:
            img, mask = sample, None

        # ensure mask exists (keep API consistent)
        if mask is None:
            mask = torch.zeros_like(img)

        # ensure correct dtype
        mask = (mask > 0.5).float()

        d = {"img": img, "mask": mask}

        if torch.max(mask) > 0:
            out = self.smart(d)
            # MONAI can return list[dict] or dict depending on version/settings
            if isinstance(out, list):
                out = out[0]
            return out["img"], out["mask"]

        out = self.fallback(d)
        return out["img"], out["mask"]




def main(args):
    misc.init_distributed_mode(args)
    print(f"Distributed: {args.distributed}")
    print('Job dir:', os.path.dirname(os.path.realpath(__file__)))
    print("Args:\n", "{}".format(args).replace(', ', ',\n'))

    args.lr = args.head_lr
    device = torch.device(args.device)

    # Repro
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.set_float32_matmul_precision('high')

    print(f"[Data] Training on patches of size {args.input_size}^3")
    transform_train = PatchCropTransform(patch_size=args.input_size)

    dataset_train = Custom3DDDatasetMask(
        args.data_path,
        transform=transform_train,
        use_lung_mask=args.use_lung_mask,
        summit_only=args.summit_only
    )

    dataset_val = None
    data_loader_val = None
    if args.val_data_path is not None:
        dataset_val = Custom3DDDatasetMask(
            args.val_data_path,
            transform=PatchCropTransform(patch_size=args.input_size),
            use_lung_mask=args.use_lung_mask
        )

    # Sampler
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if dataset_val else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None

    # Logger
    log_writer = None
    if (not args.distributed or misc.get_rank() == 0) and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # Train loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True, persistent_workers=False,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False, persistent_workers=False,
        )

    print("[model] Using simple ConvAE: MONAI AutoencoderKL via ConvAEAdapter")

    model = ConvAEAdapter(
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


    perceptual_model = None
    if args.perceptual_weight > 0:
        # Note: If you enable this, you must ensure the ViT can handle the patch size (96).
        # Standard ViT weights are for 224 or 256.
        pass

    # GAN discriminator
    discriminator = None
    disc_optimizer = None

    if args.use_gan and args.gan_weight > 0.0:
        print("[GAN] Initializing 3D PatchGAN discriminator")
        # PatchGAN adapts to input size automatically.
        # For 96^3 input, 4 layers = receptive field covers significant portion of patch.
        discriminator = PatchDiscriminator3D(
            in_channels=1,
            base_channels=64,
            n_layers=3, # Reduced layers slightly for smaller patches (optional, 4 is also fine)
        ).to(device)

        if args.optim8bit:
            disc_optimizer = OptimCls(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.9))
        else:
            disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.9))

        if args.distributed:
            discriminator = torch.nn.parallel.DistributedDataParallel(
                discriminator, device_ids=[args.gpu], find_unused_parameters=True
            )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Optimizers & Rest of script remains similar...
    conv_param_group = {
        "params": model_without_ddp.parameters(),
        "lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "base_lr": args.head_lr,
    }

    if args.optim8bit:
        optimizer = OptimCls([conv_param_group], betas=(0.9, 0.95), eps=1e-7, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([conv_param_group], betas=(0.9, 0.95), weight_decay=args.weight_decay)

    loss_scaler = NativeScaler()
    ema = EMA(model_without_ddp, decay=args.ema_decay) if args.use_ema else None

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.resume:
        print(f"[resume] Resuming full state from: {args.resume}")
        misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            load_optim=False,
            strict_model=True
        )
    else:
        print("[init] Training from scratch.")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            if data_loader_val: data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args, ema=ema,
            perceptual_model=perceptual_model,
            discriminator=discriminator, disc_optimizer=disc_optimizer
        )

        if (epoch + 1) == args.epochs and ema is not None:
            ema.copy_to(model_without_ddp)
            if hasattr(model, "module"): ema.copy_to(model.module)

        if args.output_dir and ((epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs):
            misc.save_model(
                args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler,
                discriminator=discriminator, disc_optimizer=disc_optimizer
            )

        val_stats = {}
        if data_loader_val and (((epoch + 1) % args.eval_every == 0) or ((epoch + 1) == args.epochs)):
            print(f"[eval] Running validation at epoch {epoch}")
            val_stats = eval_one_epoch(
                model, data_loader_val, device, epoch, args=args,
                perceptual_model=perceptual_model
            )

        log_stats = {
            'epoch': epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }

        if args.output_dir and misc.is_main_process():
            if log_writer: log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Training time', total_time_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Patch autoencoder training", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
