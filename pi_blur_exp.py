"""
pi_blur_exp.py — Experiment P1: Gaussian-blur exact PI caches at σ ∈ {0.5, 1, 2, 3}
and save to data/TLCGNN_blur_s{σ}/<name>.npy for downstream LP runs.

Usage:
    python pi_blur_exp.py           # generate all blurred caches
    python pi_blur_exp.py --verify  # verify existing caches without regenerating

The blurred caches are consumed via TLCGNN_PI_DIR env var in loaddatas.py:
    TLCGNN_PI_DIR=data/TLCGNN_blur_s1 python pipelines.py --datasets Chameleon ...

The dataset→filename mapping follows loaddatas.py:
    Photo     → Photo.npy
    Computers → Computers.npy
    Chameleon → chameleon.npy   (lowercase, as loaddatas uses d_name directly)
"""

import os
import sys
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

SIGMAS = [0.5, 1.0, 2.0, 3.0]

# Datasets where PDGNN beat exact PI (per spec).
# Note: loaddatas stores Chameleon as 'chameleon.npy' (lowercase).
DATASETS = {
    'Photo':     'Photo.npy',
    'Computers': 'Computers.npy',
    'Chameleon': 'chameleon.npy',
}

SOURCE_DIR = './data/TLCGNN'


def blur_pi_cache(arr, sigma):
    """
    Gaussian-blur each PI row (25-element → 5×5 image) with given sigma.
    Row ORDER is preserved; only values change.

    Args:
        arr:   np.ndarray shape [N, 25]
        sigma: float, blur kernel width in pixels

    Returns:
        blurred: np.ndarray shape [N, 25], same dtype as input
    """
    N = arr.shape[0]
    assert arr.shape[1] == 25, f"Expected 25 PI values per edge, got {arr.shape[1]}"
    # Reshape to (N, 5, 5), apply per-image blur, reshape back
    imgs = arr.reshape(N, 5, 5)
    out = np.empty_like(imgs)
    # gaussian_filter on (N,5,5) with sigma=(0, sigma, sigma) applies blur
    # only along spatial dims, not across rows
    out = gaussian_filter(imgs, sigma=(0, sigma, sigma))
    return out.reshape(N, 25)


def generate_blurred_caches(sigmas=None, verify_only=False):
    if sigmas is None:
        sigmas = SIGMAS

    for sigma in sigmas:
        # Format sigma for directory name: 0.5→s0.5, 1.0→s1, etc.
        sigma_str = str(sigma).rstrip('0').rstrip('.') if '.' in str(sigma) else str(sigma)
        # Simplify: use format that avoids trailing zeros
        sigma_str = f"{sigma:g}"
        out_dir = f'./data/TLCGNN_blur_s{sigma_str}'
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== sigma={sigma} -> {out_dir} ===")

        for dataset_name, filename in DATASETS.items():
            src_path = os.path.join(SOURCE_DIR, filename)
            dst_path = os.path.join(out_dir, filename)

            if not os.path.exists(src_path):
                print(f"  [SKIP] {dataset_name}: source not found at {src_path}")
                continue

            if verify_only:
                if os.path.exists(dst_path):
                    cached = np.load(dst_path)
                    print(f"  [OK]   {dataset_name}: {dst_path} shape={cached.shape}")
                else:
                    print(f"  [MISSING] {dataset_name}: {dst_path}")
                continue

            print(f"  Loading {dataset_name} from {src_path}...")
            arr = np.load(src_path)
            print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")

            print(f"  Applying Gaussian blur (sigma={sigma})...")
            blurred = blur_pi_cache(arr, sigma)

            # Sanity checks
            assert blurred.shape == arr.shape, f"Shape mismatch: {blurred.shape} vs {arr.shape}"
            assert not np.isnan(blurred).any(), "NaN in blurred output"
            # Blurring should not increase max by much (it normalizes, may reduce)
            print(f"  Value range: [{blurred.min():.4f}, {blurred.max():.4f}] "
                  f"(orig: [{arr.min():.4f}, {arr.max():.4f}])")

            np.save(dst_path, blurred)
            print(f"  Saved -> {dst_path}")

    print("\nDone.")


def print_sigma_labels():
    """Print the sigma → directory name mapping for use in shell scripts."""
    for sigma in SIGMAS:
        sigma_str = f"{sigma:g}"
        print(f"sigma={sigma} -> TLCGNN_blur_s{sigma_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Gaussian-blurred exact PI caches')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify that blurred caches exist (do not regenerate)')
    parser.add_argument('--sigma', type=float, nargs='+', default=None,
                        help='Sigma values to process (default: 0.5 1 2 3)')
    parser.add_argument('--labels', action='store_true',
                        help='Print sigma->dir mapping and exit')
    args = parser.parse_args()

    if args.labels:
        print_sigma_labels()
        sys.exit(0)

    generate_blurred_caches(sigmas=args.sigma, verify_only=args.verify)
