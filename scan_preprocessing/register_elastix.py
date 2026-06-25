#!/usr/bin/env python3
"""
Longitudinal CT preprocessing (baseline-anchored) using SimpleITK and Elastix.
The default pipeline performs rigid alignment followed by B-spline non-rigid
registration with the bundled Elastix parameter files.

CSV format (REQUIRED headers):
  patient,timepoint,img_path,mask_path

Outputs (separated):
  --out_img_dir/<patient>/...  (CTs)
  --out_mask_dir/<patient>/... (masks)
  --out_meta_dir/<patient>/... (json + transforms; defaults to out_img_dir if not provided)

Pipeline:
  1) Read CT (SimpleITK), resample to isotropic spacing (default 1.35mm)
  2) Get mask:
      - if provided: binarize + align + resample to iso
      - else: lungmask on iso
  3) Baseline = earliest timepoint
  4) Baseline crop_start_zyx computed from baseline mask COM
  5) Save baseline params JSON
  6) For each follow-up:
      - Elastix rigid register follow-up iso -> baseline iso (optionally with masks)
      - optionally refine with Elastix B-spline non-rigid registration
      - warp moving mask with the rigid Euler transform; optionally recompute
        the mask on the registered image
      - apply baseline crop_start_zyx -> 256^3 outputs

Elastix fallback ladder:
    If --use_elastix_masks:
      1) try -fMask + -mMask (if moving mask exists)
      2) try -fMask only
      3) try without masks
  This salvages many failures like:
    "Too many samples map outside moving image buffer"

Cleanup:
  Optionally delete ONLY intermediate .nii.gz files (never removes patient dirs).
  Elastix temp work dirs can be removed per patient unless --keep_elastix_tmp.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from lungmask import mask as lungmask

DEFAULT_RIGID_PARAMETER_FILE = Path(__file__).resolve().with_name("Par_rigid_MI.txt")
DEFAULT_BSPLINE_PARAMETER_FILE = Path(__file__).resolve().with_name("Par_bs_MI_r5.txt")


# ----------------------------
# IO + misc
# ----------------------------
def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def is_missing(s: Optional[str]) -> bool:
    if s is None:
        return True
    s = str(s).strip().strip('"').strip("'")
    return s == "" or s.upper() in {"NA", "N/A", "NONE", "NULL"}


def read_image(path: Union[str, Path]) -> sitk.Image:
    return sitk.ReadImage(str(path))


def write_image(img: sitk.Image, path: Union[str, Path]) -> None:
    ensure_dir(Path(path).parent)
    sitk.WriteImage(img, str(path))


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def spacing_tag(spacing_mm: float) -> str:
    return f"{spacing_mm:.2f}".replace(".", "p")


def sanitize_label(s: str) -> str:
    s = str(s)
    s = s.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    return s.strip("_")


# ----------------------------
# Mask helpers
# ----------------------------
def binarize_mask_any_positive(m_sitk: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(m_sitk)
    out = sitk.GetImageFromArray((arr > 0).astype(np.uint8))
    out.CopyInformation(m_sitk)
    return out


def mask_center_of_mass_zyx(mask_img: sitk.Image) -> Optional[Tuple[float, float, float]]:
    arr = sitk.GetArrayFromImage(mask_img) > 0  # Z,Y,X
    coords = np.argwhere(arr)
    if coords.size == 0:
        return None
    cz, cy, cx = coords.mean(axis=0)
    return (float(cz), float(cy), float(cx))


# ----------------------------
# Resampling
# ----------------------------
def resample_like(
    moving: sitk.Image,
    reference: sitk.Image,
    interp=sitk.sitkLinear,
    default_val: float = 0.0,
    transform: Optional[sitk.Transform] = None,
) -> sitk.Image:
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(reference)
    rf.SetInterpolator(interp)
    rf.SetDefaultPixelValue(float(default_val))
    if transform is not None:
        rf.SetTransform(transform)
    return rf.Execute(moving)


def resample_iso(
    img: sitk.Image,
    spacing_iso: float,
    interp=sitk.sitkLinear,
    default_val: float = -1000.0,
) -> sitk.Image:
    old_spacing = np.array(img.GetSpacing(), dtype=float)  # (sx,sy,sz)
    old_size = np.array(img.GetSize(), dtype=int)          # (x,y,z)
    new_spacing = np.array([spacing_iso] * 3, dtype=float)
    new_size = np.ceil(old_size * (old_spacing / new_spacing)).astype(int).tolist()

    res = sitk.ResampleImageFilter()
    res.SetInterpolator(interp)
    res.SetOutputSpacing(tuple(new_spacing))
    res.SetSize([int(x) for x in new_size])
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetDefaultPixelValue(float(default_val))
    return res.Execute(img)


# ----------------------------
# Cropping
# ----------------------------
def _shift_origin(old_origin, spacing, direction, x0, y0, z0):
    D = np.array(direction, dtype=float).reshape(3, 3)
    off_idx = np.array([x0, y0, z0], dtype=float)  # xyz
    sp = np.array(spacing, dtype=float)            # xyz
    off_phys = D @ (off_idx * sp)
    return tuple(np.array(old_origin, dtype=float) + off_phys)


def compute_crop_start_from_center(
    center_zyx: Tuple[float, float, float],
    target_zyx: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    tz, ty, tx = target_zyx
    cz, cy, cx = center_zyx
    z0 = int(round(cz - tz / 2.0))
    y0 = int(round(cy - ty / 2.0))
    x0 = int(round(cx - tx / 2.0))
    return (z0, y0, x0)


def crop_by_start_strict(
    img: sitk.Image,
    start_zyx: Tuple[int, int, int],
    target_zyx: Tuple[int, int, int],
    pad_value: float,
    out_dtype=np.float32,
) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img)  # Z,Y,X
    Z, Y, X = arr.shape
    tz, ty, tx = target_zyx
    z0, y0, x0 = start_zyx
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
            constant_values=pad_value,
        )

    if crop.shape != (tz, ty, tx):
        raise RuntimeError(f"crop shape {crop.shape} != target {(tz, ty, tx)}")

    out = sitk.GetImageFromArray(crop.astype(out_dtype))
    out.SetSpacing(img.GetSpacing())
    out.SetDirection(img.GetDirection())
    out.SetOrigin(_shift_origin(img.GetOrigin(), img.GetSpacing(), img.GetDirection(), x0, y0, z0))
    return out


# ----------------------------
# Lungmask
# ----------------------------
def run_lungmask_on_iso(img_iso: sitk.Image, device: Optional[str]) -> sitk.Image:
    use_cpu = (device is None) or (str(device).lower().startswith("cpu"))
    lm = lungmask.apply(img_iso, force_cpu=use_cpu)
    mask_arr = (np.asarray(lm) > 0).astype(np.uint8)
    m = sitk.GetImageFromArray(mask_arr)
    m.CopyInformation(img_iso)
    return sitk.Cast(m, sitk.sitkUInt8)


def get_or_compute_mask_iso(
    img_iso: sitk.Image,
    pre_mask_path: Optional[str],
    original_img: sitk.Image,
    device: Optional[str],
) -> Tuple[sitk.Image, bool]:
    if pre_mask_path and (not is_missing(pre_mask_path)) and Path(pre_mask_path).is_file():
        m = read_image(pre_mask_path)
        m = binarize_mask_any_positive(m)

        same_space = (
            np.allclose(m.GetSpacing(), original_img.GetSpacing()) and
            np.allclose(m.GetOrigin(), original_img.GetOrigin()) and
            np.allclose(m.GetDirection(), original_img.GetDirection())
        )
        m_aligned = m if same_space else resample_like(
            m, original_img, interp=sitk.sitkNearestNeighbor, default_val=0.0
        )
        mask_iso = resample_like(m_aligned, img_iso, interp=sitk.sitkNearestNeighbor, default_val=0.0)
        return sitk.Cast(mask_iso, sitk.sitkUInt8), True

    return run_lungmask_on_iso(img_iso, device), False


# ----------------------------
# CSV parsing
# ----------------------------
REQUIRED_COLS = ["patient", "timepoint", "img_path", "mask_path"]


def parse_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        headers = [h.strip() for h in reader.fieldnames]
        headers_lc = {h.lower(): h for h in headers}
        missing = [c for c in REQUIRED_COLS if c not in headers_lc]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {headers}")

        out: List[Dict[str, str]] = []
        for row in reader:
            r = {k.lower().strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            out.append(r)
        return out


def group_by_patient(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        pid = str(r["patient"]).strip()
        if is_missing(pid):
            pid = "patient0"
        groups.setdefault(pid, []).append(r)
    return groups


def sort_timepoints(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def key_fn(r):
        t = r.get("timepoint", "")
        tf = safe_float(str(t))
        return (tf,) if tf is not None else (str(t),)
    return sorted(rows, key=key_fn)


def choose_baseline_index(rows: List[Dict[str, str]]) -> int:
    return 0


# ----------------------------
# Elastix helpers
# ----------------------------
def run_subprocess(cmd: List[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_elastix_result_image(out_dir: Union[str, Path]) -> Optional[str]:
    out_dir = str(out_dir)
    candidates = [
        "result.nii.gz",
        "result.0.nii.gz",
        "result.0..nii.gz",
        "result.nii",
        "result.mhd",
        "result.0.mhd",
        "result.0..mhd",
    ]
    for name in candidates:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            return p
    try:
        for fn in os.listdir(out_dir):
            if fn.startswith("result") and fn.endswith((".nii.gz", ".nii", ".mhd")):
                return os.path.join(out_dir, fn)
    except FileNotFoundError:
        return None
    return None


def run_elastix_with_fallback(
    elastix_bin: str,
    fixed_img: str,
    moving_img: str,
    parameter_file: str,
    out_dir: str,
    threads: int,
    fixed_mask: Optional[str],
    moving_mask: Optional[str],
    use_elastix_masks: bool,
) -> Tuple[bool, str]:
    """
    Try elastix with a mask policy ladder, returning (success, mode_used).

    mode_used in:
      - "full_masks"
      - "fixed_mask_only"
      - "no_masks"
      - "failed"
    """
    base_cmd = [
        str(elastix_bin),
        "-f", str(fixed_img),
        "-m", str(moving_img),
        "-p", str(parameter_file),
        "-out", str(out_dir),
    ]
    if threads and threads > 0:
        base_cmd += ["-threads", str(int(threads))]

    attempts: List[Tuple[str, List[str]]] = []

    fixed_ok = bool(fixed_mask) and Path(str(fixed_mask)).exists()
    moving_ok = bool(moving_mask) and Path(str(moving_mask)).exists()

    if use_elastix_masks and fixed_ok:
        if moving_ok:
            attempts.append((
                "full_masks",
                base_cmd + ["-fMask", str(fixed_mask), "-mMask", str(moving_mask)],
            ))
        attempts.append((
            "fixed_mask_only",
            base_cmd + ["-fMask", str(fixed_mask)],
        ))

    attempts.append(("no_masks", base_cmd))

    last_err: Optional[Exception] = None
    for mode, cmd in attempts:
        try:
            run_subprocess(cmd)
            return True, mode
        except Exception as e:
            print(f"[WARN] Elastix attempt '{mode}' failed: {e}")
            last_err = e

    print(f"[WARN] All elastix attempts failed: {last_err}")
    return False, "failed"


# ----------------------------
# Parse Elastix EulerTransform -> SimpleITK transform
# ----------------------------
def _parse_elastix_tuple(line: str) -> List[float]:
    inside = line.strip()
    inside = inside.lstrip("(").rstrip(")")
    parts = inside.split()
    vals = []
    for p in parts[1:]:
        try:
            vals.append(float(p))
        except Exception:
            pass
    return vals


def parse_elastix_euler_transform(transform_txt: Union[str, Path]) -> sitk.Euler3DTransform:
    """
    Elastix EulerTransform rigid: 3 rotations + 3 translations, with CenterOfRotationPoint.
    """
    transform_txt = str(transform_txt)
    if not os.path.exists(transform_txt):
        raise FileNotFoundError(f"Elastix transform file not found: {transform_txt}")

    transform_name = None
    params = None
    center = None

    with open(transform_txt, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("(Transform "):
                m = re.search(r'\(Transform\s+"([^"]+)"\)', s)
                if m:
                    transform_name = m.group(1)
            elif s.startswith("(TransformParameters "):
                params = _parse_elastix_tuple(s)
            elif s.startswith("(CenterOfRotationPoint "):
                center = _parse_elastix_tuple(s)

    if transform_name is None:
        raise RuntimeError(f"Could not find (Transform ...) in {transform_txt}")
    if transform_name != "EulerTransform":
        raise RuntimeError(f"Expected EulerTransform, got {transform_name} in {transform_txt}")
    if params is None or len(params) < 6:
        raise RuntimeError(f"Could not parse 6 TransformParameters from {transform_txt}")
    if center is None or len(center) < 3:
        center = [0.0, 0.0, 0.0]

    rx, ry, rz, tx, ty, tz = params[:6]
    cx, cy, cz = center[:3]

    T = sitk.Euler3DTransform()
    T.SetCenter((float(cx), float(cy), float(cz)))
    T.SetRotation(float(rx), float(ry), float(rz))       # radians
    T.SetTranslation((float(tx), float(ty), float(tz)))  # mm
    return T


# ----------------------------
# Cleanup (delete intermediates only)
# ----------------------------
def intermediate_suffixes(spacing_mm: float) -> List[str]:
    tag = spacing_tag(spacing_mm)
    return [
        f"_ct_{tag}.nii.gz",
        f"_mask_{tag}.nii.gz",
        f"_ct_rigid_{tag}.nii.gz",
        f"_mask_rigid_{tag}.nii.gz",
        f"_ct_registered_rigid_{tag}.nii.gz",
        f"_mask_registered_rigid_{tag}.nii.gz",
        f"_ct_registered_bspline_{tag}.nii.gz",
        f"_mask_registered_bspline_{tag}.nii.gz",
    ]


def cleanup_intermediate_niis(root_patient_dir: Path, suffixes: List[str], dry_run: bool = False) -> None:
    if not root_patient_dir.is_dir():
        return
    for f in root_patient_dir.rglob("*.nii.gz"):
        if any(f.name.endswith(suf) for suf in suffixes):
            if dry_run:
                print("[DRY] would delete:", f)
            else:
                f.unlink(missing_ok=True)


# ----------------------------
# Main per-patient pipeline
# ----------------------------
def process_patient(
    patient_id: str,
    rows: List[Dict[str, str]],
    out_img_dir: str,
    out_mask_dir: str,
    out_meta_dir: str,
    spacing_iso: float,
    target_zyx: Tuple[int, int, int],
    device: Optional[str],
    write_intermediate: bool,
    redo_lungmask_after_registration: bool,
    elastix_bin: str,
    par_rigid: str,
    par_bspline: Optional[str],
    threads: int,
    elastix_tmp_root: str,
    keep_elastix_tmp: bool,
    use_elastix_masks: bool,
) -> None:
    rows_sorted = sort_timepoints(rows)
    b_idx = choose_baseline_index(rows_sorted)
    baseline_row = rows_sorted[b_idx]
    follow_rows = rows_sorted[:b_idx] + rows_sorted[b_idx + 1:]

    img_dir = Path(out_img_dir) / patient_id
    mask_dir = Path(out_mask_dir) / patient_id
    meta_dir = Path(out_meta_dir) / patient_id
    ensure_dir(img_dir)
    ensure_dir(mask_dir)
    ensure_dir(meta_dir)

    # Elastix temp dirs (per patient)
    tmp_root = Path(elastix_tmp_root) / "elastix_longitudinal" / patient_id
    tmp_inputs = tmp_root / "inputs"
    tmp_work = tmp_root / "work"
    ensure_dir(tmp_inputs)
    ensure_dir(tmp_work)

    tag = spacing_tag(spacing_iso)

    def get_paths(r: Dict[str, str]) -> Tuple[str, Optional[str]]:
        img_p = str(r.get("img_path", "")).strip()
        msk_p = str(r.get("mask_path", "")).strip()
        msk_p = None if is_missing(msk_p) else msk_p
        return img_p, msk_p

    # ---- baseline ----
    b_img_path, b_mask_path = get_paths(baseline_row)
    if is_missing(b_img_path) or (not Path(b_img_path).is_file()):
        raise FileNotFoundError(f"[{patient_id}] baseline image not found: {b_img_path}")

    b_img = read_image(b_img_path)
    b_img_iso = resample_iso(b_img, spacing_iso=spacing_iso, interp=sitk.sitkLinear, default_val=-1000.0)
    b_mask_iso, b_mask_is_provided = get_or_compute_mask_iso(b_img_iso, b_mask_path, b_img, device)

    # Save baseline iso inputs for elastix
    fixed_img_iso_path = tmp_inputs / "baseline_ct_iso.nii.gz"
    fixed_mask_iso_path = tmp_inputs / "baseline_mask_iso.nii.gz"
    write_image(b_img_iso, fixed_img_iso_path)
    write_image(sitk.Cast(b_mask_iso, sitk.sitkUInt8), fixed_mask_iso_path)

    center = mask_center_of_mass_zyx(b_mask_iso)
    if center is None:
        arr = sitk.GetArrayFromImage(b_img_iso)
        Z, Y, X = arr.shape
        center = (Z / 2.0, Y / 2.0, X / 2.0)
        print(f"[{patient_id}] [WARN] baseline mask empty; using image center.")

    crop_start_zyx = compute_crop_start_from_center(center, target_zyx)

    params = {
        "patient": patient_id,
        "spacing_iso_mm": float(spacing_iso),
        "target_zyx": list(map(int, target_zyx)),
        "baseline_timepoint": baseline_row.get("timepoint"),
        "baseline_image_path": b_img_path,
        "baseline_mask_path": b_mask_path,
        "baseline_mask_is_provided": bool(b_mask_is_provided),
        "baseline_center_zyx": [float(center[0]), float(center[1]), float(center[2])],
        "crop_start_zyx": [int(crop_start_zyx[0]), int(crop_start_zyx[1]), int(crop_start_zyx[2])],
        "elastix": {
            "par_rigid": str(par_rigid),
            "par_bspline": str(par_bspline) if par_bspline else None,
            "elastix_bin": str(elastix_bin),
        },
    }
    (meta_dir / "baseline_crop_params.json").write_text(json.dumps(params, indent=2))

    # baseline outputs
    b_img_256 = crop_by_start_strict(b_img_iso, crop_start_zyx, target_zyx, pad_value=-1000.0, out_dtype=np.float32)
    b_msk_256 = crop_by_start_strict(b_mask_iso, crop_start_zyx, target_zyx, pad_value=0.0, out_dtype=np.uint8)
    b_msk_256 = sitk.Cast(b_msk_256, sitk.sitkUInt8)

    write_image(b_img_256, img_dir / "baseline_ct_256.nii.gz")
    write_image(b_msk_256, mask_dir / "baseline_mask_256.nii.gz")

    if write_intermediate:
        write_image(b_img_iso, img_dir / f"baseline_ct_{tag}.nii.gz")
        write_image(b_mask_iso, mask_dir / f"baseline_mask_{tag}.nii.gz")

    # ---- follow-ups ----
    for j, r in enumerate(follow_rows):
        img_p, msk_p = get_paths(r)
        if is_missing(img_p) or (not Path(img_p).is_file()):
            print(f"[{patient_id}] [WARN] skipping follow-up (missing image): {img_p}")
            continue

        tp_val = r.get("timepoint", f"{j}")
        tp_label = sanitize_label(str(tp_val))

        mov = read_image(img_p)
        mov_iso = resample_iso(mov, spacing_iso=spacing_iso, interp=sitk.sitkLinear, default_val=-1000.0)
        mov_mask_iso, mov_mask_is_provided = get_or_compute_mask_iso(mov_iso, msk_p, mov, device)

        # Save moving iso input for elastix
        moving_img_iso_path = tmp_inputs / f"tp{tp_label}_ct_iso.nii.gz"
        write_image(mov_iso, moving_img_iso_path)

        # If we might use -mMask, write it now
        moving_mask_iso_path: Optional[Path] = None
        if use_elastix_masks and mov_mask_is_provided:
            moving_mask_iso_path = tmp_inputs / f"tp{tp_label}_mask_iso.nii.gz"
            write_image(sitk.Cast(mov_mask_iso, sitk.sitkUInt8), moving_mask_iso_path)

        # Elastix work dir
        rigid_out_dir = tmp_work / f"tp{tp_label}_rigid"
        ensure_dir(rigid_out_dir)

        ok, mode_used = run_elastix_with_fallback(
            elastix_bin=str(elastix_bin),
            fixed_img=str(fixed_img_iso_path),
            moving_img=str(moving_img_iso_path),
            parameter_file=str(par_rigid),
            out_dir=str(rigid_out_dir),
            threads=int(threads),
            fixed_mask=str(fixed_mask_iso_path) if Path(fixed_mask_iso_path).exists() else None,
            moving_mask=str(moving_mask_iso_path) if moving_mask_iso_path is not None else None,
            use_elastix_masks=bool(use_elastix_masks),
        )
        if not ok:
            print(f"[{patient_id}] [WARN] Elastix rigid registration failed for tp={tp_val} (all fallbacks).")
            continue
        print(f"[{patient_id}] [INFO] Elastix succeeded for tp={tp_val} using mode: {mode_used}")

        rigid_transform_txt = rigid_out_dir / "TransformParameters.0.txt"
        if not rigid_transform_txt.exists():
            print(f"[{patient_id}] [WARN] Elastix transform not found for tp={tp_val}: {rigid_transform_txt}")
            continue

        # Registered CT from elastix (already in baseline grid)
        result_img_path = find_elastix_result_image(rigid_out_dir)
        if result_img_path is None or (not Path(result_img_path).exists()):
            print(f"[{patient_id}] [WARN] Elastix registered CT not found in: {rigid_out_dir}")
            continue
        mov_iso_reg = read_image(result_img_path)

        # Create SITK Euler transform from Elastix parameters (rotation+translation only)
        try:
            T = parse_elastix_euler_transform(rigid_transform_txt)
        except Exception as e:
            print(f"[{patient_id}] [WARN] Failed to parse Elastix transform for tp={tp_val}: {e}")
            T = None

        # Registered mask after rigid alignment.
        if (not mov_mask_is_provided) and redo_lungmask_after_registration:
            mov_msk_reg = run_lungmask_on_iso(mov_iso_reg, device)
        else:
            if T is None:
                mov_msk_reg = sitk.Cast(mov_mask_iso, sitk.sitkUInt8)
            else:
                mov_msk_reg = resample_like(
                    sitk.Cast(mov_mask_iso, sitk.sitkUInt8),
                    b_img_iso,
                    interp=sitk.sitkNearestNeighbor,
                    default_val=0.0,
                    transform=T,
                )
                mov_msk_reg = sitk.Cast(binarize_mask_any_positive(mov_msk_reg), sitk.sitkUInt8)

        final_transform_txt = rigid_transform_txt
        final_stage_name = "rigid"

        if par_bspline:
            bspline_out_dir = tmp_work / f"tp{tp_label}_bspline"
            ensure_dir(bspline_out_dir)

            bspline_moving_mask_path: Optional[Path] = None
            if use_elastix_masks:
                bspline_moving_mask_path = tmp_inputs / f"tp{tp_label}_mask_rigid_for_bspline.nii.gz"
                write_image(sitk.Cast(mov_msk_reg, sitk.sitkUInt8), bspline_moving_mask_path)

            ok_bs, bs_mode = run_elastix_with_fallback(
                elastix_bin=str(elastix_bin),
                fixed_img=str(fixed_img_iso_path),
                moving_img=str(result_img_path),
                parameter_file=str(par_bspline),
                out_dir=str(bspline_out_dir),
                threads=int(threads),
                fixed_mask=str(fixed_mask_iso_path) if Path(fixed_mask_iso_path).exists() else None,
                moving_mask=str(bspline_moving_mask_path) if bspline_moving_mask_path is not None else None,
                use_elastix_masks=bool(use_elastix_masks),
            )
            if ok_bs:
                bspline_result_img_path = find_elastix_result_image(bspline_out_dir)
                bspline_transform_txt = bspline_out_dir / "TransformParameters.0.txt"
                if bspline_result_img_path is not None and Path(bspline_result_img_path).exists():
                    mov_iso_reg = read_image(bspline_result_img_path)
                    final_stage_name = "bspline"
                    if bspline_transform_txt.exists():
                        final_transform_txt = bspline_transform_txt
                    if redo_lungmask_after_registration:
                        mov_msk_reg = run_lungmask_on_iso(mov_iso_reg, device)
                    print(f"[{patient_id}] [INFO] Elastix B-spline succeeded for tp={tp_val} using mode: {bs_mode}")
                else:
                    print(f"[{patient_id}] [WARN] B-spline result not found; keeping rigid result for tp={tp_val}.")
            else:
                print(f"[{patient_id}] [WARN] B-spline registration failed; keeping rigid result for tp={tp_val}.")

        # Crop outputs
        mov_256 = crop_by_start_strict(
            mov_iso_reg, crop_start_zyx, target_zyx, pad_value=-1000.0, out_dtype=np.float32
        )
        msk_256 = crop_by_start_strict(
            mov_msk_reg, crop_start_zyx, target_zyx, pad_value=0.0, out_dtype=np.uint8
        )
        msk_256 = sitk.Cast(msk_256, sitk.sitkUInt8)

        write_image(mov_256, img_dir / f"tp{tp_label}_ct_256.nii.gz")
        write_image(msk_256, mask_dir / f"tp{tp_label}_mask_256.nii.gz")

        # Save transform parameters to meta_dir (so you have it even if tmp gets cleaned)
        shutil.copy2(
            str(rigid_transform_txt),
            str(meta_dir / f"tp{tp_label}_rigid_to_baseline_TransformParameters.0.txt"),
        )
        if final_stage_name == "bspline" and final_transform_txt.exists():
            shutil.copy2(
                str(final_transform_txt),
                str(meta_dir / f"tp{tp_label}_bspline_to_baseline_TransformParameters.0.txt"),
            )

        if write_intermediate:
            write_image(mov_iso, img_dir / f"tp{tp_label}_ct_{tag}.nii.gz")
            write_image(mov_mask_iso, mask_dir / f"tp{tp_label}_mask_{tag}.nii.gz")
            write_image(mov_iso_reg, img_dir / f"tp{tp_label}_ct_registered_{final_stage_name}_{tag}.nii.gz")
            write_image(mov_msk_reg, mask_dir / f"tp{tp_label}_mask_registered_{final_stage_name}_{tag}.nii.gz")

    # Cleanup elastix temp per patient (optional)
    if not keep_elastix_tmp:
        try:
            shutil.rmtree(str(tmp_root), ignore_errors=True)
        except Exception:
            pass

    print(f"[{patient_id}] done.")
    print(f"  CTs  : {img_dir}")
    print(f"  Masks: {mask_dir}")
    print(f"  Meta : {meta_dir}")


# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(
        description="Longitudinal preprocessing: resample + mask + Elastix rigid/B-spline registration to baseline + 256^3 crop."
    )
    p.add_argument("--csv", required=True, help="CSV with headers: patient,timepoint,img_path,mask_path")

    p.add_argument("--out_img_dir", required=True, help="Output directory root for CTs (one folder per patient).")
    p.add_argument("--out_mask_dir", required=True, help="Output directory root for masks (one folder per patient).")
    p.add_argument(
        "--out_meta_dir",
        default=None,
        help="Optional output directory root for json/transforms (one folder per patient). Default: out_img_dir.",
    )

    p.add_argument("--spacing", type=float, default=1.35, help="Isotropic spacing in mm (default 1.35).")
    p.add_argument("--target", type=int, nargs=3, default=[256, 256, 256], help="Target cube size (Z Y X).")
    p.add_argument("--device", default=None, help="e.g., 'cuda:0' or 'cpu' for lungmask.")

    p.add_argument("--write_intermediate", action="store_true", help="Write intermediate isotropic and registered files.")
    p.add_argument(
        "--cleanup_intermediate",
        action="store_true",
        help="Delete only intermediate .nii.gz after each patient (both img+mask dirs).",
    )
    p.add_argument(
        "--redo_lungmask_after_registration",
        action="store_true",
        help="If set, rerun lungmask on each registered follow-up image.",
    )
    p.add_argument(
        "--dry_cleanup",
        action="store_true",
        help="If set with --cleanup_intermediate, print what would be deleted without deleting.",
    )

    # Elastix config (rigid only)
    p.add_argument(
        "--par_rigid",
        default=str(DEFAULT_RIGID_PARAMETER_FILE),
        help="Elastix rigid parameter file (EulerTransform). Defaults to scan_preprocessing/Par_rigid_MI.txt.",
    )
    p.add_argument(
        "--par_bspline",
        default=str(DEFAULT_BSPLINE_PARAMETER_FILE),
        help="Elastix B-spline parameter file. Defaults to scan_preprocessing/Par_bs_MI_r5.txt.",
    )
    p.add_argument("--no_bspline", action="store_true", help="Disable the B-spline non-rigid refinement stage.")
    p.add_argument("--elastix_bin", default="elastix", help="Path to elastix binary (or rely on PATH).")
    p.add_argument("--threads", type=int, default=8, help="Threads for elastix (-threads).")
    p.add_argument(
        "--elastix_tmp_root",
        default=None,
        help="Root for elastix temp dirs. Default: $TMPDIR if set, else /tmp",
    )
    p.add_argument("--keep_elastix_tmp", action="store_true", help="Keep elastix temp work dirs per patient.")
    p.add_argument(
        "--use_elastix_masks",
        action="store_true",
        help="If set, pass -fMask and -mMask to elastix when available (recommended).",
    )

    args = p.parse_args()
    if not Path(args.par_rigid).is_file():
        raise FileNotFoundError(f"Elastix rigid parameter file not found: {args.par_rigid}")
    par_bspline = None if args.no_bspline else args.par_bspline
    if par_bspline is not None and not Path(par_bspline).is_file():
        raise FileNotFoundError(f"Elastix B-spline parameter file not found: {par_bspline}")

    # Keep elastix itself in control of its threading; avoid extra ITK threads in our process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

    ensure_dir(args.out_img_dir)
    ensure_dir(args.out_mask_dir)
    out_meta_dir = args.out_meta_dir if args.out_meta_dir else args.out_img_dir
    ensure_dir(out_meta_dir)

    elastix_tmp_root = args.elastix_tmp_root
    if elastix_tmp_root is None:
        elastix_tmp_root = os.environ.get("TMPDIR", "/tmp")
    ensure_dir(elastix_tmp_root)

    rows = parse_rows(args.csv)
    if not rows:
        raise RuntimeError("CSV is empty.")

    groups = group_by_patient(rows)
    suf = intermediate_suffixes(float(args.spacing))

    for pid, pid_rows in groups.items():
        img_patient_dir = Path(args.out_img_dir) / str(pid)
        mask_patient_dir = Path(args.out_mask_dir) / str(pid)
        try:
            process_patient(
                patient_id=str(pid),
                rows=pid_rows,
                out_img_dir=args.out_img_dir,
                out_mask_dir=args.out_mask_dir,
                out_meta_dir=out_meta_dir,
                spacing_iso=float(args.spacing),
                target_zyx=(int(args.target[0]), int(args.target[1]), int(args.target[2])),
                device=args.device,
                write_intermediate=bool(args.write_intermediate),
                redo_lungmask_after_registration=bool(args.redo_lungmask_after_registration),
                elastix_bin=args.elastix_bin,
                par_rigid=args.par_rigid,
                par_bspline=par_bspline,
                threads=int(args.threads),
                elastix_tmp_root=str(elastix_tmp_root),
                keep_elastix_tmp=bool(args.keep_elastix_tmp),
                use_elastix_masks=bool(args.use_elastix_masks),
            )
            if args.cleanup_intermediate:
                cleanup_intermediate_niis(img_patient_dir, suffixes=suf, dry_run=bool(args.dry_cleanup))
                cleanup_intermediate_niis(mask_patient_dir, suffixes=suf, dry_run=bool(args.dry_cleanup))
        except Exception as e:
            print(f"[ERROR] patient {pid}: {e}")


if __name__ == "__main__":
    main()
