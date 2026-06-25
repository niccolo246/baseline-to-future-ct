#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
from lungmask import mask as lungmask
import csv


def binarize_12_numpy(m_sitk: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(m_sitk)
    bin_arr = ((arr == 1) | (arr == 2)).astype(np.uint8)  # lungs -> 1, background -> 0
    out = sitk.GetImageFromArray(bin_arr)
    out.CopyInformation(m_sitk)
    return out


def print_mask_stats(mask_img, name="mask"):
    arr = sitk.GetArrayFromImage(mask_img)
    total = arr.size
    zeros = np.count_nonzero(arr == 0)
    nonzeros = total - zeros
    frac_nonzero = nonzeros / total
    uniq = np.unique(arr)

    print(f"--- {name} ---")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Unique values: {uniq.tolist()}")
    print(f"Zero voxels: {zeros} ({zeros/total*100:.2f}%)")
    print(f"Non-zero voxels: {nonzeros} ({frac_nonzero*100:.2f}%)")
    print(f"Fraction non-zero: {frac_nonzero:.4f}")
    print()
    return frac_nonzero


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def read_image(path: str) -> sitk.Image:
    return sitk.ReadImage(path)

def write_image(img: sitk.Image, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    sitk.WriteImage(img, path)

def stem_nii(path: str) -> str:
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem

def is_missing(s: Optional[str]) -> bool:
    if s is None:
        return True
    s = str(s).strip().strip('"').strip("'")
    return (s == "" or s.upper() in {"NA", "N/A", "NONE", "NULL"})

def resample_like(moving: sitk.Image, reference: sitk.Image,
                  interp=sitk.sitkLinear, default_val: float = 0.0) -> sitk.Image:
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(reference)
    rf.SetInterpolator(interp)
    rf.SetDefaultPixelValue(float(default_val))
    return rf.Execute(moving)


# ----------------------------
# Resampling
# ----------------------------
def resample_iso(img: sitk.Image,
                 spacing_iso: float,
                 interp=sitk.sitkLinear,
                 default_hu: float = -1000.0) -> sitk.Image:
    old_spacing = np.array(img.GetSpacing(), dtype=float)
    old_size    = np.array(img.GetSize(), dtype=int)
    new_spacing = np.array([spacing_iso] * 3, dtype=float)
    new_size    = np.ceil(old_size * (old_spacing / new_spacing)).astype(int).tolist()

    res = sitk.ResampleImageFilter()
    res.SetInterpolator(interp)
    res.SetOutputSpacing(tuple(new_spacing))
    res.SetSize([int(x) for x in new_size])
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetDefaultPixelValue(float(default_hu))
    return res.Execute(img)


# ----------------------------
# Mask + BBox + COM
# ----------------------------
def bbox_from_mask(mask_img: sitk.Image) -> Optional[Tuple[int, int, int, int, int, int]]:
    arr = sitk.GetArrayFromImage(mask_img)
    coords = np.where(arr > 0)
    if coords[0].size == 0:
        return None
    zmin, zmax = int(coords[0].min()), int(coords[0].max())
    ymin, ymax = int(coords[1].min()), int(coords[1].max())
    xmin, xmax = int(coords[2].min()), int(coords[2].max())
    return (zmin, zmax, ymin, ymax, xmin, xmax)

def mask_center_of_mass_zyx(mask_img: sitk.Image) -> Optional[Tuple[float, float, float]]:
    arr = sitk.GetArrayFromImage(mask_img) > 0
    coords = np.argwhere(arr)
    if coords.size == 0:
        return None
    cz, cy, cx = coords.mean(axis=0)
    return (float(cz), float(cy), float(cx))

def dilate_bbox_mm(bbox, spacing_xyz, margin_mm=(20, 20, 30)):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    sx, sy, sz = spacing_xyz  # (sx, sy, sz)
    mx = int(round(margin_mm[0] / sx))
    my = int(round(margin_mm[1] / sy))
    mz = int(round(margin_mm[2] / sz))
    return (zmin - mz, zmax + mz, ymin - my, ymax + my, xmin - mx, xmax + mx)

def clamp_bbox(bbox, shape_zyx):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    Z, Y, X = shape_zyx
    zmin = max(0, zmin); ymin = max(0, ymin); xmin = max(0, xmin)
    zmax = min(Z - 1, zmax); ymax = min(Y - 1, ymax); xmax = min(X - 1, xmax)
    return (zmin, zmax, ymin, ymax, xmin, xmax)

def _bbox_extents_vox(b):
    zmin, zmax, ymin, ymax, xmin, xmax = b
    return (zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1)

def bbox_center_zyx(bbox) -> Tuple[float, float, float]:
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    cz = (zmin + zmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx = (xmin + xmax) / 2.0
    return (float(cz), float(cy), float(cx))

def choose_crop_center(mask_img: sitk.Image,
                       center_mode: str = "bbox",
                       crop_margin_mm: Tuple[float, float, float] = (20, 20, 30)) -> Tuple[Tuple[float, float, float], Optional[Tuple[int, int, int, int, int, int]]]:
    """
    Returns:
        center_zyx
        bbox_used_for_centering (dilated/clamped bbox if bbox mode was possible, else None)
    """
    arr = sitk.GetArrayFromImage(mask_img)
    Z, Y, X = arr.shape

    bbox = bbox_from_mask(mask_img)
    com = mask_center_of_mass_zyx(mask_img)

    if center_mode == "bbox" and bbox is not None:
        sx, sy, sz = mask_img.GetSpacing()
        bbox_d = dilate_bbox_mm(bbox, (sx, sy, sz), margin_mm=crop_margin_mm)
        bbox_d = clamp_bbox(bbox_d, (Z, Y, X))
        center = bbox_center_zyx(bbox_d)
        return center, bbox_d

    if center_mode == "com" and com is not None:
        return com, None

    if center_mode == "bbox" and bbox is None and com is not None:
        print("[WARN] bbox center requested but no valid mask bbox found; falling back to COM.")
        return com, None

    if center_mode == "com" and com is None and bbox is not None:
        print("[WARN] COM center requested but COM unavailable; falling back to bbox center.")
        sx, sy, sz = mask_img.GetSpacing()
        bbox_d = dilate_bbox_mm(bbox, (sx, sy, sz), margin_mm=crop_margin_mm)
        bbox_d = clamp_bbox(bbox_d, (Z, Y, X))
        center = bbox_center_zyx(bbox_d)
        return center, bbox_d

    print("[WARN] No valid mask found; using image center for cropping.")
    return (Z / 2.0, Y / 2.0, X / 2.0), None


# ----------------------------
# Crop (STRICT center with padding)
# ----------------------------
def _shift_origin(old_origin, spacing, direction, x0, y0, z0):
    """
    Shift the physical origin by an index offset (x0,y0,z0) in voxel units.
    """
    D = np.array(direction, dtype=float).reshape(3, 3)
    off_idx = np.array([x0, y0, z0], dtype=float)
    sp = np.array(spacing, dtype=float)
    off_phys = D @ (off_idx * sp)
    return tuple(np.array(old_origin, dtype=float) + off_phys)

def crop_center_box_strict(img: sitk.Image,
                           center_zyx: Tuple[float, float, float],
                           target=(256, 256, 256),
                           pad_value=-1000.0) -> sitk.Image:
    """
    Crop a fixed target cube centered at center_zyx (in array index coords Z,Y,X).
    If the crop extends outside the image, pad to preserve centering.
    """
    arr = sitk.GetArrayFromImage(img)  # Z,Y,X
    Z, Y, X = arr.shape
    tz, ty, tx = target
    cz, cy, cx = center_zyx

    z0 = int(round(cz - tz / 2.0))
    y0 = int(round(cy - ty / 2.0))
    x0 = int(round(cx - tx / 2.0))

    z1, y1, x1 = z0 + tz, y0 + ty, x0 + tx

    iz0, iy0, ix0 = max(0, z0), max(0, y0), max(0, x0)
    iz1, iy1, ix1 = min(Z, z1), min(Y, y1), min(X, x1)

    crop = arr[iz0:iz1, iy0:iy1, ix0:ix1]

    pzb = max(0, -z0); pza = max(0, z1 - Z)
    pyb = max(0, -y0); pya = max(0, y1 - Y)
    pxb = max(0, -x0); pxa = max(0, x1 - X)

    if any(v > 0 for v in (pzb, pza, pyb, pya, pxb, pxa)):
        crop = np.pad(
            crop,
            ((pzb, pza), (pyb, pya), (pxb, pxa)),
            mode="constant",
            constant_values=pad_value
        )

    if crop.shape != (tz, ty, tx):
        raise RuntimeError(f"Internal error: crop has shape {crop.shape}, expected {(tz, ty, tx)}")

    out = sitk.GetImageFromArray(crop.astype(np.float32))
    out.SetSpacing(img.GetSpacing())
    out.SetDirection(img.GetDirection())
    out.SetOrigin(_shift_origin(img.GetOrigin(), img.GetSpacing(), img.GetDirection(), x0, y0, z0))
    return out


# ----------------------------
# Pipeline per case
# ----------------------------
def process_case(input_path: str,
                 out_img_path: str,
                 out_mask_path: str,
                 spacing_iso: float = 1.4,
                 device: Optional[str] = None,
                 crop_target: Optional[Tuple[int, int, int]] = None,
                 crop_margin_mm: Tuple[float, float, float] = (20, 20, 30),
                 center_mode: str = "bbox",
                 pre_mask_path: Optional[str] = None) -> None:
    # Step 1: Read image
    img = read_image(input_path)

    # Step 2: If a pre-existing mask is provided, read & align to the image grid
    pre_mask_img = None
    if pre_mask_path and not is_missing(pre_mask_path) and Path(pre_mask_path).is_file():
        m = read_image(pre_mask_path)
        m = binarize_12_numpy(m)

        same_space = (np.allclose(m.GetSpacing(), img.GetSpacing()) and
                      np.allclose(m.GetOrigin(),  img.GetOrigin())  and
                      np.allclose(m.GetDirection(), img.GetDirection()))
        pre_mask_img = m if same_space else resample_like(m, img, interp=sitk.sitkNearestNeighbor, default_val=0.0)

    # Step 3: Isotropic resample the image
    img_r = resample_iso(img, spacing_iso=spacing_iso, interp=sitk.sitkLinear, default_hu=-1000.0)

    # Step 4: Get mask (reuse or compute), then resample it to match img_r
    if pre_mask_img is not None:
        mask_iso = resample_like(pre_mask_img, img_r, interp=sitk.sitkNearestNeighbor, default_val=0.0)
    else:
        use_cpu = (device is None) or (str(device).lower().startswith("cpu"))
        lm = lungmask.apply(img_r, force_cpu=use_cpu)
        mask_arr = (np.asarray(lm) > 0).astype(np.uint8)
        mask_iso = sitk.GetImageFromArray(mask_arr)
        mask_iso.CopyInformation(img_r)

    final_img, final_mask = img_r, mask_iso

    if crop_target is not None:
        center, bbox_used = choose_crop_center(
            mask_iso,
            center_mode=center_mode,
            crop_margin_mm=crop_margin_mm
        )

        # Overflow warning based on dilated bbox if available; otherwise raw bbox
        bbox = bbox_used if bbox_used is not None else bbox_from_mask(mask_iso)
        if bbox is not None:
            if bbox_used is None:
                sx, sy, sz = img_r.GetSpacing()
                bbox = dilate_bbox_mm(bbox, (sx, sy, sz), margin_mm=crop_margin_mm)
                Z, Y, X = sitk.GetArrayFromImage(img_r).shape
                bbox = clamp_bbox(bbox, (Z, Y, X))

            bz, by, bx = _bbox_extents_vox(bbox)
            tz, ty, tx = crop_target
            if (bz > tz) or (by > ty) or (bx > tx):
                print(f"[WARN] overflow: bbox {bx}x{by}x{bz} > target {tx}x{ty}x{tz} at spacing {img_r.GetSpacing()}")

        final_img = crop_center_box_strict(img_r, center, target=crop_target, pad_value=-1000.0)
        final_mask = crop_center_box_strict(mask_iso, center, target=crop_target, pad_value=0.0)
    else:
        print("[INFO] Crop not enabled; only isotropic resampling done.")

    final_mask = sitk.Cast(final_mask, sitk.sitkUInt8)
    write_image(final_img, out_img_path)
    write_image(final_mask, out_mask_path)

    if crop_target is not None:
        assert final_img.GetSize() == tuple(crop_target), f"Got {final_img.GetSize()} not {crop_target}"
        assert final_mask.GetSize() == tuple(crop_target), f"Mask {final_mask.GetSize()} not {crop_target}"


# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(
        description="Resample CT to fixed isotropic spacing and segment lungs; optionally crop to fixed cube centered on lungs."
    )
    p.add_argument("--input", required=True,
                   help="Path to .nii/.nii.gz, or a .txt with one path per line, or a .csv with Image Path/Mask Path.")
    p.add_argument("--out_dir", default='outputs/preprocessed/img',
                   help="Output directory.")
    p.add_argument("--spacing", type=float, default=1.35,
                   help="Isotropic spacing in mm.")
    p.add_argument("--device", default=None,
                   help="e.g., 'cuda:0' or 'cpu'. Default: CPU.")
    p.add_argument("--crop", action="store_true",
                   help="Also crop to a fixed cube centered on lungs.")
    p.add_argument("--target", type=int, nargs=3, default=[256, 256, 256],
                   help="Crop target size, e.g., 256 256 256.")
    p.add_argument("--margin", type=float, nargs=3, default=[10, 10, 10],
                   help="BBox dilation margin in mm (mx my mz).")
    p.add_argument("--center_mode", choices=["bbox", "com"], default="bbox",
                   help="How to choose crop center: 'bbox' uses center of dilated lung bbox; 'com' uses lung mask center of mass.")
    p.add_argument("--mask_out_dir", default='outputs/preprocessed/mask',
                   help="Optional separate output directory for masks. If not set, masks go to --out_dir.")
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "3")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

    ensure_dir(args.out_dir)
    if args.mask_out_dir:
        ensure_dir(args.mask_out_dir)

    def run_one(in_path: str, pre_mask_path: Optional[str] = None):
        st = stem_nii(in_path)
        out_img = str(Path(args.out_dir) / f"{st}_resampled.nii.gz")
        mask_dir = Path(args.mask_out_dir) if args.mask_out_dir else Path(args.out_dir)
        out_msk = str(mask_dir / f"{st}_lungmask.nii.gz")


        if Path(out_img).is_file() and Path(out_msk).is_file():
            print(f"[SKIP] {in_path} -> outputs already exist")
            return

        try:
            process_case(
                input_path=in_path,
                out_img_path=out_img,
                out_mask_path=out_msk,
                spacing_iso=args.spacing,
                device=args.device,
                crop_target=tuple(args.target) if args.crop else None,
                crop_margin_mm=tuple(args.margin),
                center_mode=args.center_mode,
                pre_mask_path=pre_mask_path
            )
            print(f"[OK] {in_path} -> {out_img} & {out_msk} (mask_src={'pre' if pre_mask_path else 'lungmask'}, center_mode={args.center_mode})")
        except Exception as e:
            print(f"[ERROR] {in_path}: {e}")

    if args.input.endswith(".csv"):
        with open(args.input, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_p = (row.get("Image Path") or row.get("img_path") or row.get("image") or "").strip()
                msk_p = (row.get("Mask Path") or row.get("mask_path") or row.get("mask") or "").strip()
                run_one(img_p, None if is_missing(msk_p) or not Path(msk_p).is_file() else msk_p)

    elif args.input.endswith(".txt"):
        lines = [l.strip() for l in Path(args.input).read_text().splitlines() if l.strip()]
        for pth in lines:
            run_one(pth, None)

    else:
        run_one(args.input, None)


if __name__ == "__main__":
    main()
