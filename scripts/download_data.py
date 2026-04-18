"""
Download all D4RL datasets needed for multi-environment pretraining.
Run this ONCE before starting training to avoid mid-run download failures.

Usage:
    python tabrl/scripts/download_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tabrl.data.d4rl_loader import ALL_PRETRAIN_ENVS

def download_all():
    try:
        import minari
    except ImportError:
        print("ERROR: pip install minari")
        sys.exit(1)

    print(f"Downloading {len(ALL_PRETRAIN_ENVS)} datasets ...\n")
    print("Dataset IDs:")
    for mid in ALL_PRETRAIN_ENVS:
        print(f"  {mid}")
    print()

    failed = []
    for mid in ALL_PRETRAIN_ENVS:
        print(f"  Downloading {mid} ...")
        try:
            minari.download_dataset(mid)
            print(f"  ✓ {mid}")
        except Exception as e:
            print(f"  ✗ {mid}  ERROR: {e}")
            failed.append(mid)

    print(f"\n{'='*50}")
    if failed:
        print(f"  {len(failed)} downloads failed:")
        for f in failed:
            print(f"    {f}")
        sys.exit(1)
    else:
        print(f"  All {len(ALL_PRETRAIN_ENVS)} datasets downloaded successfully.")
        print(f"  Ready to train: python tabrl/train.py --phase pretrain")

if __name__ == "__main__":
    download_all()