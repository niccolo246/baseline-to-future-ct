# ct_networks.py
# Python 3.9-compatible, robust loading, and safer context handling.

import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet, ControlNet

# -------------------------------------------------------------------------
# Global CT config
# -------------------------------------------------------------------------

INPUT_SIZE = 256

# Latent space
LATENT_CHANNELS = 4
LATENT_DIMS = (64, 64, 64)

# Fallback defaults (main should pass context_dim dynamically)
CT_CONTEXT_DIM = 7 #17
CT_EMBEDDING_DIM = 128  # project conditioning scalars -> 128 dim

# Safe attention head size (recommended by MONAI)
# Must have same length as num_channels
NUM_HEAD_CHANNELS = (0, 64, 64)

# UNet backbone config (keep centralized so ControlNet matches)
UNET_NUM_CHANNELS = (128, 256, 512)
UNET_ATTENTION_LEVELS = (False, False, True)
UNET_NUM_RES_BLOCKS = 2
UNET_NORM_GROUPS = 32
UNET_NORM_EPS = 1e-6
UNET_TRANSFORMER_LAYERS = 1


# -------------------------------------------------------------------------
# Scalar conditioning embedding projector
# -------------------------------------------------------------------------

class ScalarEmbedder(nn.Module):
    """
    Projects raw conditioning scalars (B,1,dim) into (B,1,CT_EMBEDDING_DIM).
    A small 2-layer MLP with SiLU.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected: (B,1,input_dim) OR (B,input_dim)
        if x is None:
            return None
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B,1,input_dim)
        return self.model(x)


# -------------------------------------------------------------------------
# Wrapper around MONAI DiffusionModelUNet
# -------------------------------------------------------------------------

class ProjectedDiffusionWrapper(nn.Module):
    """
    Wraps UNet so that context scalars are projected by ScalarEmbedder
    before cross-attention.
    """
    def __init__(self, unet: nn.Module, embedder: nn.Module):
        super().__init__()
        self.unet = unet
        self.embedder = embedder

    def forward(self, x, timesteps, context=None, **kwargs):
        if context is not None:
            context = self.embedder(context)  # (B,1,embed_dim)
        return self.unet(x=x, timesteps=timesteps, context=context, **kwargs)


# -------------------------------------------------------------------------
# Wrapper for ControlNet
# -------------------------------------------------------------------------

class ProjectedControlNetWrapper(nn.Module):
    """
    Wraps ControlNet to apply scalar embedding before controlnet forward pass.
    """
    def __init__(self, controlnet: nn.Module, embedder: nn.Module):
        super().__init__()
        self.controlnet = controlnet
        self.embedder = embedder

    def forward(self, x, timesteps, context=None, controlnet_cond=None, **kwargs):
        if context is not None:
            context = self.embedder(context)
        return self.controlnet(
            x=x,
            timesteps=timesteps,
            context=context,
            controlnet_cond=controlnet_cond,
            **kwargs
        )


# -------------------------------------------------------------------------
# Latent Diffusion UNet Constructor
# -------------------------------------------------------------------------

def init_latent_diffusion_ct(checkpoints_path=None, args=None, context_dim=None) -> nn.Module:
    """
    Builds the latent UNet for CT + scalar conditioning.

    Args:
        checkpoints_path: optional path to .pth to load weights
        args: argparse namespace (expects .conditioning_mode)
        context_dim: number of input context features (covariates + mask bit).
    """
    conditioning_mode = getattr(args, "conditioning_mode", "full")

    if context_dim is None:
        context_dim = CT_CONTEXT_DIM

    if conditioning_mode == "none":
        unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=LATENT_CHANNELS,
            out_channels=LATENT_CHANNELS,
            num_res_blocks=UNET_NUM_RES_BLOCKS,
            num_channels=UNET_NUM_CHANNELS,
            attention_levels=UNET_ATTENTION_LEVELS,
            norm_num_groups=UNET_NORM_GROUPS,
            norm_eps=UNET_NORM_EPS,
            resblock_updown=True,
            num_head_channels=NUM_HEAD_CHANNELS,
            transformer_num_layers=UNET_TRANSFORMER_LAYERS,
            with_conditioning=False,
            cross_attention_dim=None,
            upcast_attention=False,
            use_flash_attention=False
        )
        model: nn.Module = unet
    else:
        unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=LATENT_CHANNELS,
            out_channels=LATENT_CHANNELS,
            num_res_blocks=UNET_NUM_RES_BLOCKS,
            num_channels=UNET_NUM_CHANNELS,
            attention_levels=UNET_ATTENTION_LEVELS,
            norm_num_groups=UNET_NORM_GROUPS,
            norm_eps=UNET_NORM_EPS,
            resblock_updown=True,
            num_head_channels=NUM_HEAD_CHANNELS,
            transformer_num_layers=UNET_TRANSFORMER_LAYERS,
            with_conditioning=True,
            cross_attention_dim=CT_EMBEDDING_DIM,
            upcast_attention=False,
            use_flash_attention=False,
        )
        embedder = ScalarEmbedder(context_dim, CT_EMBEDDING_DIM)
        model = ProjectedDiffusionWrapper(unet, embedder)

    # ----------------------------
    # Smart Weight Loading
    # ----------------------------
    if checkpoints_path is not None:
        print(f"[latent_diffusion_ct] Loading weights from: {checkpoints_path}")
        ckpt = torch.load(checkpoints_path, map_location="cpu")

        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]

        model_sd = model.state_dict()
        model_keys = set(model_sd.keys())

        new_state_dict = {}
        mapped = 0
        direct = 0

        for k, v in ckpt.items():
            if k in model_keys:
                new_state_dict[k] = v
                direct += 1
            else:
                # Handle checkpoint from bare UNet loaded into wrapper
                k2 = f"unet.{k}"
                if k2 in model_keys:
                    new_state_dict[k2] = v
                    mapped += 1

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"  Loaded keys: direct={direct}, mapped_to_unet={mapped}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys (ignored): {len(unexpected)}")

        # Helpful sanity warning
        if conditioning_mode != "none":
            # If UNet conv_in missing, prefix mapping might be wrong
            if any("unet.conv_in" in k for k in missing):
                print("WARNING: Some core UNet weights are missing (e.g., unet.conv_in.*). Check checkpoint key prefixes.")
            # If embedder missing, that's expected if checkpoint didn't have it
            if any("embedder" in k for k in missing):
                print("Note: embedder weights missing (expected if training from scratch / new context_dim).")

    return model


# -------------------------------------------------------------------------
# ControlNet Constructor (Optional)
# -------------------------------------------------------------------------

def init_controlnet_ct(
    checkpoints_path=None,
    context_dim=None,
    conditioning_embedding_in_channels=None,
) -> nn.Module:
    """
    Builds ControlNet + scalar embedder.

    Notes:
      conditioning_embedding_in_channels must match the channel dimension of
      controlnet_cond you will pass at runtime.

    Common setups:
      - mask-only: 1
      - latent-only: LATENT_CHANNELS (4)
      - latent+mask: LATENT_CHANNELS + 1 (5)

    Args:
        checkpoints_path: optional path to .pth
        context_dim: number of scalar context features (covariates + mask token)
        conditioning_embedding_in_channels: override channels for controlnet_cond
    """
    if context_dim is None:
        context_dim = CT_CONTEXT_DIM

    if conditioning_embedding_in_channels is None:
        # Default to latent + time-channel conditioning.
        conditioning_embedding_in_channels = LATENT_CHANNELS + 1

    controlnet = ControlNet(
        spatial_dims=3,
        in_channels=LATENT_CHANNELS,
        num_res_blocks=UNET_NUM_RES_BLOCKS,
        num_channels=UNET_NUM_CHANNELS,
        attention_levels=UNET_ATTENTION_LEVELS,
        norm_num_groups=UNET_NORM_GROUPS,
        norm_eps=UNET_NORM_EPS,
        resblock_updown=True,
        num_head_channels=NUM_HEAD_CHANNELS,
        transformer_num_layers=UNET_TRANSFORMER_LAYERS,
        with_conditioning=True,
        cross_attention_dim=CT_EMBEDDING_DIM,
        upcast_attention=False,
        use_flash_attention=False,
        conditioning_embedding_in_channels=conditioning_embedding_in_channels,
        conditioning_embedding_num_channels=(256,),
    )

    embedder = ScalarEmbedder(context_dim, CT_EMBEDDING_DIM)
    model = ProjectedControlNetWrapper(controlnet, embedder)

    if checkpoints_path is not None:
        print(f"[controlnet_ct] Loading weights from: {checkpoints_path}")
        ckpt = torch.load(checkpoints_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]

        model_sd = model.state_dict()
        model_keys = set(model_sd.keys())

        new_sd = {}
        direct = 0
        mapped = 0

        def strip_module(k):
            return k.replace("module.", "", 1) if k.startswith("module.") else k

        for k, v in ckpt.items():
            k = strip_module(k)
            if k in model_keys:
                new_sd[k] = v
                direct += 1
            else:
                k2 = f"controlnet.{k}"
                if k2 in model_keys:
                    new_sd[k2] = v
                    mapped += 1

        incompatible = model.load_state_dict(new_sd, strict=False)
        print(f"  Loaded keys: direct={direct}, mapped_to_controlnet={mapped}")
        print(f"  Missing keys: {len(incompatible.missing_keys)}")
        print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")

    return model
