# From Baseline to Future CT

Code for the MICCAI paper **"From Baseline to Future CT: Large-Scale Diffusion Pretraining for Single-Scan IPF Progression Prediction"**.

This repository contains the minimum paper pipeline for the proposed model:

1. Preprocess thoracic CTs to isotropic 1.35 mm, fixed 256^3 volumes, and lung masks.
2. Train the patch autoencoder on 96^3 lung-biased patches.
3. Encode full 256^3 scans into stitched 4 x 64^3 latents.
4. Pretrain the covariate-conditioned latent diffusion model on screening CT latents.
5. Pretrain the longitudinal ControlNet adaptor on screening baseline-follow-up pairs.
6. Fine-tune the ControlNet adaptor on IPF pairs with lung-biased patch-decoded supervision.
7. Generate future CT predictions with DDIM sampling and EMA ControlNet weights.

No data, trained model checkpoints, or metrics/baseline-comparison scripts are included.

## Installation

Use Python 3.9+ with a CUDA-enabled PyTorch install. Then install the remaining direct dependencies:

```bash
pip install -r requirements.txt
```

The code vendors the small subset of MONAI GenerativeModels used by the paper under `generative/`.
Optional acceleration packages such as `bitsandbytes` or `xformers` can be installed separately if you choose to enable those code paths.

## Weights

This repository does not release trained autoencoder, latent-diffusion, or ControlNet checkpoints. Any `--checkpoint`, `--ae_ckpt`, `--diff_ckpt`, `--cnet_ckpt`, or `--cnet_ckpt_init` argument should point to weights you trained or otherwise have permission to use.

The optional CT foundation encoder used for perceptual-feature evaluation is separate from the proposed generative model weights. If you need those foundation-model weights for a perceptual metric workflow, they are available from [Zenodo](https://zenodo.org/records/18835750).

## Paper Defaults

The command-line defaults match the proposed-model implementation details in the manuscript:

| Stage | Script | LR | Batch | Epochs |
| --- | --- | ---: | ---: | ---: |
| Patch autoencoder | `main_pretrain.py` | `1e-4` | `6` | `120` |
| Latent diffusion | `ct_generative_modelling/main_latent_diffusion_ct.py` | `3e-5` | `16` | `100` |
| Screening ControlNet | `ct_generative_modelling/main_ct_controlnet.py` | `6e-5` | `4` | `50` |
| IPF ControlNet | `ct_generative_modelling/main_controlnet_finetune_ipf.py` | `4e-6` | `4` | `30` |

Other paper defaults are also set in code: 256^3 CT grid, 4 x 64^3 latent grid, 96^3 AE patches, 25% overlap, DDPM `T=1000`, DDIM `300`, latent-average `m=5` at inference, and proposed-model `lambda_GAN=0` for IPF ControlNet fine-tuning.

## Data Format

The scripts expect de-identified data paths supplied by the user.

For latent diffusion CSVs, the default dataset expects:

```text
latent_img_path, latent_mask_path, z_age, sex_male, z_height_cm, z_bmi, z_pack_years, z_log1p_lung_vol
```

For ControlNet pair CSVs, the default dataset expects:

```text
patient, timepoint, latent_img_path, latent_mask_path,
z_age, sex_male, z_height_cm, z_bmi, z_pack_years, z_log1p_lung_vol
```

`timepoint` should be in years. Pairs are constructed with `0 <= delta_t <= 3`.

## Pipeline

### 1. Preprocess CTs

For paper-faithful preprocessing, provide lung masks computed externally (the manuscript used TotalSegmentator) in a CSV with image and mask paths. If no mask is supplied, the script falls back to `lungmask` for convenience.

```bash
python scan_preprocessing/scan_downsample_crop256.py \
  --input data/ct_paths_and_masks.csv \
  --out_dir outputs/preprocessed/img \
  --mask_out_dir outputs/preprocessed/mask \
  --spacing 1.35 \
  --crop \
  --target 256 256 256
```

Convert preprocessed images to `.npz` for patch autoencoder training:

```bash
python scan_preprocessing/nii_to_npz.py \
  --input data/preprocessed_img_paths.txt
```

For longitudinal IPF pairs, use the bundled Elastix rigid + B-spline configurations to register follow-up scans to baseline before latent extraction:

```bash
python scan_preprocessing/register_elastix.py \
  --csv data/ipf_longitudinal_scans.csv \
  --out_img_dir outputs/registered/img \
  --out_mask_dir outputs/registered/mask \
  --spacing 1.35 \
  --target 256 256 256 \
  --use_elastix_masks
```

The default registration parameter files are `scan_preprocessing/Par_rigid_MI.txt` and `scan_preprocessing/Par_bs_MI_r5.txt`. You can override them with `--par_rigid` and `--par_bspline`, or disable the B-spline stage with `--no_bspline`.

### 2. Train Patch Autoencoder

```bash
python main_pretrain.py \
  --data_path data/ae_train_paths.txt \
  --output_dir outputs/autoencoder \
  --log_dir outputs/autoencoder
```

### 3. Extract Full-Volume Latents

```bash
python create_latents.py \
  --data_list data/registered_ct_paths.txt \
  --checkpoint "$AE_CKPT" \
  --output_dir outputs/latents \
  --latent_list_out outputs/latents/latent_paths.txt \
  --mask_list_out outputs/latents/mask_paths.txt
```

### 4. Pretrain Latent Diffusion

```bash
python ct_generative_modelling/main_latent_diffusion_ct.py \
  --data_path data/screening_latents_train.csv \
  --val_data_path data/screening_latents_val.csv \
  --ae_ckpt "$AE_CKPT" \
  --output_dir outputs/latent_diffusion \
  --log_dir outputs/latent_diffusion
```

### 5. Pretrain Screening ControlNet

```bash
python ct_generative_modelling/main_ct_controlnet.py \
  --train_csv data/screening_pairs_train.csv \
  --val_csv data/screening_pairs_val.csv \
  --ae_ckpt "$AE_CKPT" \
  --diff_ckpt "$LDM_CKPT" \
  --output_dir outputs/controlnet_pretrain \
  --log_dir outputs/controlnet_pretrain
```

### 6. Fine-Tune IPF ControlNet

```bash
python ct_generative_modelling/main_controlnet_finetune_ipf.py \
  --train_csv data/ipf_pairs_train.csv \
  --val_csv data/ipf_pairs_val.csv \
  --ae_ckpt "$AE_CKPT" \
  --diff_ckpt "$LDM_CKPT" \
  --cnet_ckpt_init "$SCREENING_CNET_CKPT" \
  --output_dir outputs/controlnet_ipf \
  --log_dir outputs/controlnet_ipf
```

The proposed model leaves the optional adversarial switch off and uses `--gan_weight 0.0`.

### 7. Generate Future CTs

```bash
python ct_generative_modelling/infer_progression.py \
  --val_csv data/ipf_pairs_test.csv \
  --ae_ckpt "$AE_CKPT" \
  --diff_ckpt "$LDM_CKPT" \
  --cnet_ckpt "$IPF_CNET_CKPT" \
  --output_dir outputs/predictions \
  --latent_average_samples 5
```

Outputs are saved as NIfTI volumes under `outputs/predictions/`.

## Repository Layout

```text
main_pretrain.py                         # patch autoencoder training
create_latents.py                        # full-volume latent extraction
datasets_three_d.py                      # latent and pair datasets
ct_generative_modelling/
  networks_shallow.py                    # diffusion and ControlNet definitions
  main_latent_diffusion_ct.py            # latent diffusion training
  main_ct_controlnet.py                  # screening ControlNet pretraining
  main_controlnet_finetune_ipf.py        # IPF ControlNet adaptation
  infer_progression.py                   # proposed-model inference
scan_preprocessing/                      # CT resampling, cropping, registration
generative/                              # vendored MONAI generative components
util/                                    # scheduler and distributed helpers
```

## Citation

Please cite the MICCAI paper if you use this code. A BibTeX entry can be added here once the proceedings metadata is available.
