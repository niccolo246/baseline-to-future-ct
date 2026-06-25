#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import re


def preprocess_ct_hu_to_unit(vol: np.ndarray) -> np.ndarray:
    """
    Same CT intensity preprocessing used by the AE dataset:
      - nan/inf cleaning
      - clip to [-1200, 800]
      - normalize to [0,1] via (x + 1200) / 2000
    """
    vol = np.nan_to_num(vol, nan=-1200.0, posinf=800.0, neginf=-1200.0)
    np.clip(vol, -1200.0, 800.0, out=vol)
    vol = (vol + 1200.0) / 2000.0
    np.clip(vol, 0.0, 1.0, out=vol)
    return vol.astype(np.float32)


def nii_to_npz(
    nii_path: str,
    npz_path: str,
    to_float16: bool = True,
    report_size: bool = False,
):
    """
    Convert a .nii / .nii.gz file to compressed .npz:
      - 'data': normalized [0,1] volume (float16 or float32)
      - 'affine': canonical affine (float32)
      - 'orig_dtype': original NIfTI dtype (string)
    """
    nii_path = str(nii_path)
    npz_path = str(npz_path)

    # For size reporting
    orig_size = os.path.getsize(nii_path) if os.path.exists(nii_path) else None

    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)

    vol = img.get_fdata(dtype=np.float32)
    orig_dtype = str(img.get_data_dtype())

    # Match training preprocessing
    vol = preprocess_ct_hu_to_unit(vol)  # float32, [0,1]

    if to_float16:
        vol = vol.astype(np.float16)

    affine = img.affine.astype(np.float32)

    np.savez_compressed(
        npz_path,
        data=vol,
        affine=affine,
        orig_dtype=orig_dtype,
    )

    new_size = os.path.getsize(npz_path)
    msg = (
        f"[OK] {nii_path} -> {npz_path} | "
        f"shape={vol.shape}, dtype={vol.dtype}"
    )
    if report_size and orig_size is not None:
        ratio = new_size / orig_size if orig_size > 0 else float('nan')
        msg += f" | {orig_size/1e6:.2f} MB -> {new_size/1e6:.2f} MB (ratio={ratio:.3f})"
    print(msg)


def process_list(list_path: str, delete_nii: bool = False,
                 to_float16: bool = True, report_size: bool = False):
    """
    Process all paths in a text file (one path per line).
    For each .nii / .nii.gz:
      - create .npz next to it (same basename)
      - optionally delete original NIfTI
    """
    list_path = Path(list_path)
    assert list_path.exists(), f"List file not found: {list_path}"

    with list_path.open("r") as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Found {len(paths)} paths in {list_path}")

    for nii_path in paths:
        # Skip empty / commented lines
        if not nii_path or nii_path.startswith("#"):
            continue

        p = Path(nii_path)
        if not p.exists():
            print(f"[WARN] File does not exist, skipping: {p}")
            continue

        # Only handle .nii / .nii.gz
        if not (str(p).endswith(".nii") or str(p).endswith(".nii.gz")):
            print(f"[WARN] Not a NIfTI file, skipping: {p}")
            continue

        # Build .npz path with same basename
        # e.g. foo.nii.gz -> foo.npz, foo.nii -> foo.npz
        npz_path = re.sub(r"\.nii(\.gz)?$", ".npz", str(p))

        try:
            nii_to_npz(
                nii_path=str(p),
                npz_path=npz_path,
                to_float16=to_float16,
                report_size=report_size,
            )
            if delete_nii:
                os.remove(p)
                print(f"[INFO] Deleted original NIfTI: {p}")
        except Exception as e:
            print(f"[ERROR] Failed on {p}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a list of NIfTI (.nii/.nii.gz) to .npz with CT preprocessing."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Text file with one NIfTI path per line (this slice of the array job).",
    )
    parser.add_argument(
        "--delete_nii",
        action="store_true",
        help="If set, delete the original NIfTI after successful conversion.",
    )
    parser.add_argument(
        "--no_float16",
        action="store_true",
        help="If set, keep data as float32 instead of float16.",
    )
    parser.add_argument(
        "--report_size",
        action="store_true",
        help="If set, print before/after file sizes and compression ratio.",
    )

    args = parser.parse_args()
    to_float16 = not args.no_float16

    process_list(
        list_path=args.input,
        delete_nii=args.delete_nii,
        to_float16=to_float16,
        report_size=args.report_size,
    )


if __name__ == "__main__":
    main()


