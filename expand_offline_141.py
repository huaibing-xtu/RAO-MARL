"""
Expand the case141 offline dataset: poor 200→300, medium 500→700, then re-mix to 1200.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import glob

import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_eps(root: str):
    return sorted(glob.glob(os.path.join(root, "ep_*.npz")))


def run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)


def collect_additional(split_name, extra_episodes, start_day, args, temp_dir):
    """Collect additional episodes to a temp dir."""
    ensure_dir(temp_dir)
    cmd = [
        sys.executable,
        "offline_safe/dataset/build_dataset.py",
        "--scenario", args.scenario,
        "--mode", args.mode,
        "--voltage-barrier-type", args.voltage_barrier_type,
        "--episode-limit", str(args.episode_limit),
        "--episodes", str(extra_episodes),
        "--seed", str(args.seed),
        "--policy", "random" if split_name == "poor" else "noisy_droop",
        "--save-dir", temp_dir,
        "--start-day", str(start_day),
        "--day-step", str(args.day_step),
        "--hour", str(args.hour),
        "--quarter", str(args.quarter),
    ]
    if args.manual_reset:
        cmd.append("--manual-reset")
    if split_name == "medium":
        cmd += ["--droop-gain", str(args.droop_gain), "--droop-noise-std", str(args.droop_noise_std)]
    run_cmd(cmd)


def copy_with_offset(src_dir, dst_dir, offset):
    """Copy ep_XXXXX.npz from src_dir to dst_dir with index offset."""
    ensure_dir(dst_dir)
    src_files = list_eps(src_dir)
    for fp in src_files:
        old_idx = int(os.path.basename(fp).replace("ep_", "").replace(".npz", ""))
        new_idx = old_idx + offset
        dst_fp = os.path.join(dst_dir, f"ep_{new_idx:05d}.npz")
        shutil.copy2(fp, dst_fp)
        print(f"  Copied {os.path.basename(fp)} -> {os.path.basename(dst_fp)}")
    print(f"  Total: {len(src_files)} files copied to {dst_dir}")


def summarize_dataset_dir(dataset_dir, split_name="replay"):
    files = list_eps(dataset_dir)
    if len(files) == 0:
        raise ValueError(f"{split_name} dir empty: {dataset_dir}")
    episode_returns, episode_costs, avg_v_outs, sum_destroys, lengths = [], [], [], [], []
    for fp in files:
        d = np.load(fp)
        episode_returns.append(float(d["rewards"].sum()))
        episode_costs.append(float(d["costs"].sum()))
        avg_v_outs.append(float(d["v_out"].mean()) if "v_out" in d else 0.0)
        sum_destroys.append(float(d["destroy"].sum()) if "destroy" in d else 0.0)
        lengths.append(int(len(d["rewards"])))
    return {
        "split": split_name,
        "num_episodes": len(files),
        "avg_return": float(np.mean(episode_returns)),
        "avg_cost": float(np.mean(episode_costs)),
        "avg_v_out": float(np.mean(avg_v_outs)),
        "avg_destroy": float(np.mean(sum_destroys)),
        "avg_length": float(np.mean(lengths)),
    }


def mix_replay(total_episodes, poor_dir, medium_dir, good_dir, replay_dir, ratios, seed):
    ensure_dir(replay_dir)
    poor_files = list_eps(poor_dir)
    medium_files = list_eps(medium_dir)
    good_files = list_eps(good_dir)
    assert len(poor_files) > 0, f"poor dir empty: {poor_dir}"
    assert len(medium_files) > 0, f"medium dir empty: {medium_dir}"
    assert len(good_files) > 0, f"good dir empty: {good_dir}"

    rng = random.Random(seed)
    poor_ratio, medium_ratio, good_ratio = ratios
    n_poor = int(total_episodes * poor_ratio)
    n_medium = int(total_episodes * medium_ratio)
    n_good = total_episodes - n_poor - n_medium

    assert len(poor_files) >= n_poor, f"Not enough poor: need {n_poor}, have {len(poor_files)}"
    assert len(medium_files) >= n_medium, f"Not enough medium: need {n_medium}, have {len(medium_files)}"
    assert len(good_files) >= n_good, f"Not enough good: need {n_good}, have {len(good_files)}"

    for fp in list_eps(replay_dir):
        os.remove(fp)

    rng.shuffle(poor_files)
    rng.shuffle(medium_files)
    rng.shuffle(good_files)

    selected = []
    selected += [("poor", fp) for fp in poor_files[:n_poor]]
    selected += [("medium", fp) for fp in medium_files[:n_medium]]
    selected += [("good", fp) for fp in good_files[:n_good]]
    rng.shuffle(selected)

    for idx, (src_type, src_fp) in enumerate(selected):
        dst_fp = os.path.join(replay_dir, f"ep_{idx:05d}.npz")
        shutil.copy2(src_fp, dst_fp)

    print(f"\nMixed replay: {len(selected)} episodes (poor={n_poor}, medium={n_medium}, good={n_good})")
    return summarize_dataset_dir(replay_dir, "replay")


def main():
    parser = argparse.ArgumentParser(description="Expand case141 offline dataset and re-mix.")
    parser.add_argument("--scenario", type=str, default="case141_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--manual-reset", action="store_true", default=True)
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=0)
    parser.add_argument("--quarter", type=int, default=0)

    parser.add_argument("--dataset-root", type=str, default="./offline_safe/data/offline_case141_all")
    parser.add_argument("--alg", type=str, default="maddpg")

    parser.add_argument("--poor-episodes", type=int, default=200)
    parser.add_argument("--medium-episodes", type=int, default=500)
    parser.add_argument("--replay-total", type=int, default=1200)
    parser.add_argument("--poor-ratio", type=float, default=0.1)
    parser.add_argument("--medium-ratio", type=float, default=0.5)

    parser.add_argument("--droop-gain", type=float, default=2.0)
    parser.add_argument("--droop-noise-std", type=float, default=0.02)

    parser.add_argument("--skip-poor", action="store_true")
    parser.add_argument("--skip-medium", action="store_true")
    parser.add_argument("--skip-mix", action="store_true")
    args = parser.parse_args()

    model_root = os.path.join(args.dataset_root, args.alg)
    poor_dir = os.path.join(model_root, "poor")
    medium_dir = os.path.join(model_root, "medium")
    good_dir = os.path.join(model_root, "good")
    replay_dir = os.path.join(model_root, "replay")
    temp_dir = os.path.join(model_root, "_temp_expand")

    good_ratio = round(1.0 - args.poor_ratio - args.medium_ratio, 2)
    ratios = (args.poor_ratio, args.medium_ratio, good_ratio)

    # Count existing
    n_poor_existing = len(list_eps(poor_dir))
    n_medium_existing = len(list_eps(medium_dir))
    n_good_existing = len(list_eps(good_dir))
    print(f"Existing: poor={n_poor_existing}, medium={n_medium_existing}, good={n_good_existing}")
    print(f"Target: poor={args.poor_episodes}, medium={args.medium_episodes}")
    print(f"Replay total: {args.replay_total}, ratios: {ratios}")

    # --- Expand poor ---
    if not args.skip_poor:
        n_extra_poor = args.poor_episodes - n_poor_existing
        if n_extra_poor > 0:
            print(f"\n[1/3] Collecting +{n_extra_poor} poor episodes (start_day={n_poor_existing})...")
            temp_poor = os.path.join(temp_dir, "poor")
            if os.path.exists(temp_poor):
                shutil.rmtree(temp_poor)
            collect_additional("poor", n_extra_poor, n_poor_existing, args, temp_poor)
            copy_with_offset(temp_poor, poor_dir, n_poor_existing)
            print(f"  poor now has {len(list_eps(poor_dir))} episodes")
        else:
            print(f"\n[1/3] Poor already has {n_poor_existing} >= {args.poor_episodes}, skipping")
    else:
        print(f"\n[1/3] Skipping poor expansion (--skip-poor)")

    # --- Expand medium ---
    if not args.skip_medium:
        n_extra_medium = args.medium_episodes - n_medium_existing
        if n_extra_medium > 0:
            print(f"\n[2/3] Collecting +{n_extra_medium} medium episodes (start_day={n_medium_existing})...")
            temp_medium = os.path.join(temp_dir, "medium")
            if os.path.exists(temp_medium):
                shutil.rmtree(temp_medium)
            collect_additional("medium", n_extra_medium, n_medium_existing, args, temp_medium)
            copy_with_offset(temp_medium, medium_dir, n_medium_existing)
            print(f"  medium now has {len(list_eps(medium_dir))} episodes")
        else:
            print(f"\n[2/3] Medium already has {n_medium_existing} >= {args.medium_episodes}, skipping")
    else:
        print(f"\n[2/3] Skipping medium expansion (--skip-medium)")

    # --- Cleanup temp ---
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp dir")

    # --- Re-mix ---
    if not args.skip_mix:
        print(f"\n[3/3] Mixing replay ({args.replay_total} total)...")
        n_poor_final = len(list_eps(poor_dir))
        n_medium_final = len(list_eps(medium_dir))
        n_good_final = len(list_eps(good_dir))
        print(f"  Available: poor={n_poor_final}, medium={n_medium_final}, good={n_good_final}")

        replay_summary = mix_replay(
            total_episodes=args.replay_total,
            poor_dir=poor_dir,
            medium_dir=medium_dir,
            good_dir=good_dir,
            replay_dir=replay_dir,
            ratios=ratios,
            seed=args.seed,
        )

        # Save summaries
        for split_name, d in [("poor", poor_dir), ("medium", medium_dir), ("good", good_dir), ("replay", replay_dir)]:
            if split_name == "replay":
                summary = replay_summary
            else:
                summary = summarize_dataset_dir(d, split_name)
            with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  {split_name}: {summary['num_episodes']} eps, ret={summary['avg_return']:.2f}, cost={summary['avg_cost']:.2f}, v_out={summary['avg_v_out']:.4f}")

        all_summary = {
            "alg": args.alg,
            "ratios": {"poor": args.poor_ratio, "medium": args.medium_ratio, "good": good_ratio},
            "replay_total": args.replay_total,
            "poor": summarize_dataset_dir(poor_dir, "poor"),
            "medium": summarize_dataset_dir(medium_dir, "medium"),
            "good": summarize_dataset_dir(good_dir, "good"),
            "replay": replay_summary,
        }
        all_summary_path = os.path.join(model_root, "all_summary.json")
        with open(all_summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summary, f, indent=2, ensure_ascii=False)
        print(f"\nAll summary saved to {all_summary_path}")
    else:
        print(f"\n[3/3] Skipping mix (--skip-mix)")

    print("\nDone.")


if __name__ == "__main__":
    main()
