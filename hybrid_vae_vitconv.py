import torch
import torch.nn as nn
import torch.nn.functional as F

from generative.networks.nets.autoencoderkl import AutoencoderKL


class ConvAEAdapter(nn.Module):
    """Small adapter around the MONAI AutoencoderKL used in the paper pipeline."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_channels=(16, 64, 96, 128),
        num_res_blocks=(2, 3, 3, 4),
        latent_channels=32,
        attention_levels=(False, False, False, False),
        norm_num_groups=16,
        norm_eps=1e-6,
        use_flash_attention=False,
        use_checkpointing=False,
        use_convtranspose=False,
        output_sigmoid=True,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        logvar_min=-12.0,
        logvar_max=6.0,
    ):
        super().__init__()
        self.ae = AutoencoderKL(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            attention_levels=attention_levels,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            use_checkpointing=use_checkpointing,
            use_convtranspose=use_convtranspose,
        )
        self.output_sigmoid = output_sigmoid
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    @staticmethod
    def _rsample_from_mu_logvar(mu, logvar, use_mean: bool):
        if use_mean:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask_ratio: float = 0.0, use_mean: bool = False):
        mu, sigma = self.ae.encode(x)
        logvar = 2.0 * torch.log(sigma.clamp_min(1e-6))
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

        z = self._rsample_from_mu_logvar(mu, logvar, use_mean=use_mean)
        y256 = self.ae.decode(z)
        if self.output_sigmoid:
            y256 = torch.sigmoid(y256)

        y128 = F.avg_pool3d(y256, kernel_size=2, stride=2)
        return y256, y128, mu, logvar


class PatchDiscriminator3D(nn.Module):
    """Lightweight 3D PatchGAN-style discriminator for optional AE/IPF losses."""

    def __init__(self, in_channels=1, base_channels=32, n_layers=4):
        super().__init__()
        layers = [
            nn.Sequential(
                nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]

        ch = base_channels
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(base_channels * 2 ** i, 256)
            layers.append(
                nn.Sequential(
                    nn.Conv3d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm3d(ch, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        layers.append(nn.Conv3d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
