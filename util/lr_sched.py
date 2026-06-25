# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """
    Cosine schedule with warmup that preserves per-group base_lr.

    We compute a unit scale s in [0, 1], then set:
        lr_group = base_lr_group * ((1 - min_frac) * s + min_frac)

    Where:
      - During warmup, s linearly increases 0 -> 1.
      - After warmup, s follows a cosine decay 1 -> 0 over the remaining epochs.
      - min_frac is computed from args.min_lr / args.lr so that args.min_lr
        acts like a global floor fraction (preserves LLRD ratios).

    This yields:
      epoch at warmup start   -> lr ≈ base_lr_group * min_frac
      epoch at warmup end     -> lr ≈ base_lr_group
      final epoch             -> lr ≈ base_lr_group * min_frac
    """
    # warmup progress (0..1 over warmup_epochs)
    if getattr(args, "warmup_epochs", 0) > 0 and epoch < args.warmup_epochs:
        s = max(0.0, min(1.0, epoch / float(args.warmup_epochs)))
    else:
        # cosine over the remaining epochs
        total = max(1, getattr(args, "epochs", 1) - max(0, getattr(args, "warmup_epochs", 0)))
        t = min(1.0, max(0.0, (epoch - getattr(args, "warmup_epochs", 0)) / float(total)))
        # classic half-cycle: 1 -> 0 (decay)
        s = 0.5 * (1.0 + math.cos(math.pi * t))
        # NOTE: no inversion here; we want decay from high -> low

    # interpret min_lr as a fraction of the global lr (preserves group ratios)
    min_frac = 0.0
    global_lr = getattr(args, "lr", None)
    min_lr = getattr(args, "min_lr", None)
    if global_lr is not None and global_lr > 0 and min_lr is not None:
        min_frac = max(0.0, min(1.0, min_lr / float(global_lr)))

    last_lr = None
    for pg in optimizer.param_groups:
        base = pg.get("base_lr", pg.get("lr", 0.0))
        lr = base * ((1.0 - min_frac) * s + min_frac)
        pg["lr"] = lr
        last_lr = lr

    return last_lr if last_lr is not None else 0.0
