# offline_safe/dataset/mix_replay.py
import os
import glob
import json
import shutil
import random
import argparse
import numpy as np


def list_eps(root):
    return sorted(glob.glob(os.path.join(root, "ep_*.npz")))


def sample_without_replacement(files, n, rng):
    if n > len(files):
        raise ValueError(f"Need {n} files, but only {len(files)} available in {files[:1]}")
    idx = list(range(len(files)))
    rng.shuffle(idx)
    return [files[i] for i in idx[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poor-dir", type=str, required=True)
    parser.add_argument("--medium-dir", type=str, required=True)
    parser.add_argument("--good-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)

    parser.add_argument("--total-episodes", type=int, default=300)
    parser.add_argument("--poor-ratio", type=float, default=0.3)
    parser.add_argument("--medium-ratio", type=float, default=0.4)
    parser.add_argument("--good-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    rng = random.Random(args.seed)

    poor_files = list_eps(args.poor_dir)
    medium_files = list_eps(args.medium_dir)
    good_files = list_eps(args.good_dir)

    n_poor = int(args.total_episodes * args.poor_ratio)
    n_medium = int(args.total_episodes * args.medium_ratio)
    n_good = args.total_episodes - n_poor - n_medium

    def episode_score(fp, alpha=5.0):
        with np.load(fp) as d:
            ep_return = float(d["rewards"].sum())
            ep_cost = float(d["costs"].sum())
        return ep_return - alpha * ep_cost

    all_files = poor_files + medium_files + good_files
    scored = [(fp, episode_score(fp)) for fp in all_files]
    scored.sort(key=lambda x: x[1])

    n = len(scored)
    poor_pool = [fp for fp, _ in scored[: n // 3]]
    medium_pool = [fp for fp, _ in scored[n // 3: 2 * n // 3]]
    good_pool = [fp for fp, _ in scored[2 * n // 3:]]

    selected = []
    selected += [("poor", fp) for fp in sample_without_replacement(poor_pool, n_poor, rng)]
    selected += [("medium", fp) for fp in sample_without_replacement(medium_pool, n_medium, rng)]
    selected += [("good", fp) for fp in sample_without_replacement(good_pool, n_good, rng)]

    rng.shuffle(selected)

    manifest = []
    for new_idx, (src_type, src_fp) in enumerate(selected):
        dst_fp = os.path.join(args.save_dir, f"ep_{new_idx:05d}.npz")
        shutil.copy2(src_fp, dst_fp)
        manifest.append({
            "dst": os.path.basename(dst_fp),
            "src_type": src_type,
            "src_path": src_fp,
        })

    with open(os.path.join(args.save_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_episodes": args.total_episodes,
            "poor_ratio": args.poor_ratio,
            "medium_ratio": args.medium_ratio,
            "good_ratio": args.good_ratio,
            "manifest": manifest,
        }, f, indent=2)

    print(f"Replay dataset saved to: {args.save_dir}")


if __name__ == "__main__":
    main()