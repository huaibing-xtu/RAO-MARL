"""
case141 混合质量离线数据集构建脚本。
按照 case33 的 mixed replay 流程：
  poor   (10%): 随机策略
  medium (50%): noisy droop 策略
  good   (40%): expert checkpoint 策略
三者混合后写入 replay/ 目录供 V3 训练使用。
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

DEFAULT_CHECKPOINT = "./model_save/var_voltage_control-case141_3min_final-distributed-maddpg-l1-1/model.pt"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def list_eps(root: str):
    return sorted(glob.glob(os.path.join(root, "ep_*.npz")))


def run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)


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


def collect_split(split_name, episodes, args, save_dir):
    """Call build_dataset.py to collect one quality split."""
    ensure_dir(save_dir)

    cmd = [
        sys.executable,
        "offline_safe/dataset/build_dataset.py",
        "--scenario", args.scenario,
        "--mode", args.mode,
        "--voltage-barrier-type", args.voltage_barrier_type,
        "--episode-limit", str(args.episode_limit),
        "--episodes", str(episodes),
        "--seed", str(args.seed),
        "--policy", split_name if split_name == "random" else "noisy_droop",
        "--save-dir", save_dir,
        "--start-day", str(args.start_day),
        "--day-step", str(args.day_step),
        "--hour", str(args.hour),
        "--quarter", str(args.quarter),
    ]
    if args.manual_reset:
        cmd.append("--manual-reset")
    if split_name == "noisy_droop":
        cmd += ["--droop-gain", str(args.droop_gain), "--droop-noise-std", str(args.droop_noise_std)]

    run_cmd(cmd)
    return summarize_dataset_dir(save_dir, split_name)


def mix_replay(total_episodes, poor_dir, medium_dir, good_dir, replay_dir, ratios, seed):
    ensure_dir(replay_dir)
    poor_files = list_eps(poor_dir)
    medium_files = list_eps(medium_dir)
    good_files = list_eps(good_dir)
    if len(poor_files) == 0:
        raise ValueError(f"poor dir empty: {poor_dir}")
    if len(medium_files) == 0:
        raise ValueError(f"medium dir empty: {medium_dir}")
    if len(good_files) == 0:
        raise ValueError(f"good dir empty: {good_dir}")

    rng = random.Random(seed)
    poor_ratio, medium_ratio, good_ratio = ratios
    n_poor = int(total_episodes * poor_ratio)
    n_medium = int(total_episodes * medium_ratio)
    n_good = total_episodes - n_poor - n_medium

    if len(poor_files) < n_poor:
        raise ValueError(f"Not enough poor: need {n_poor}, have {len(poor_files)}")
    if len(medium_files) < n_medium:
        raise ValueError(f"Not enough medium: need {n_medium}, have {len(medium_files)}")
    if len(good_files) < n_good:
        raise ValueError(f"Not enough good: need {n_good}, have {len(good_files)}")

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
    parser = argparse.ArgumentParser(description="Build mixed-quality offline dataset for case141.")
    parser.add_argument("--scenario", type=str, default="case141_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=0)
    parser.add_argument("--quarter", type=int, default=0)

    parser.add_argument("--dataset-root", type=str, default="./offline_safe/data/offline_case141_all")
    parser.add_argument("--alg", type=str, default="maddpg")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)

    parser.add_argument("--poor-episodes", type=int, default=200)
    parser.add_argument("--medium-episodes", type=int, default=500)
    parser.add_argument("--replay-total", type=int, default=800)
    parser.add_argument("--poor-ratio", type=float, default=0.1)
    parser.add_argument("--medium-ratio", type=float, default=0.5)

    parser.add_argument("--droop-gain", type=float, default=4.0)
    parser.add_argument("--droop-noise-std", type=float, default=0.02)

    parser.add_argument("--skip-poor", action="store_true", help="Skip poor collection (use existing)")
    parser.add_argument("--skip-medium", action="store_true", help="Skip medium collection (use existing)")
    parser.add_argument("--skip-good", action="store_true", help="Skip good collection (use existing)")
    parser.add_argument("--skip-mix", action="store_true", help="Skip mixing step")
    args = parser.parse_args()

    model_root = os.path.join(args.dataset_root, args.alg)
    poor_dir = os.path.join(model_root, "poor")
    medium_dir = os.path.join(model_root, "medium")
    good_dir = os.path.join(model_root, "good")
    replay_dir = os.path.join(model_root, "replay")
    ensure_dir(model_root)

    good_ratio = round(1.0 - args.poor_ratio - args.medium_ratio, 2)
    ratios = (args.poor_ratio, args.medium_ratio, good_ratio)

    print("=" * 60)
    print(f"Mixed replay builder for case141")
    print(f"  Dataset root: {args.dataset_root}")
    print(f"  Algorithm: {args.alg}")
    print(f"  Ratios: poor={args.poor_ratio}, medium={args.medium_ratio}, good={good_ratio}")
    print(f"  Replay total: {args.replay_total}")
    print("=" * 60)

    # --- Collect poor (random) ---
    if not args.skip_poor:
        print(f"\n[1/4] Collecting poor (random) data: {args.poor_episodes} episodes ...")
        poor_summary = collect_split("random", args.poor_episodes, args, poor_dir)
        save_json(os.path.join(poor_dir, "summary.json"), poor_summary)
        print(f"  poor: {poor_summary['num_episodes']} eps, avg_return={poor_summary['avg_return']:.2f}, avg_cost={poor_summary['avg_cost']:.2f}")
    else:
        print(f"\n[1/4] Skipping poor collection (--skip-poor)")

    # --- Collect medium (noisy droop) ---
    if not args.skip_medium:
        print(f"\n[2/4] Collecting medium (noisy_droop) data: {args.medium_episodes} episodes ...")
        medium_summary = collect_split("noisy_droop", args.medium_episodes, args, medium_dir)
        save_json(os.path.join(medium_dir, "summary.json"), medium_summary)
        print(f"  medium: {medium_summary['num_episodes']} eps, avg_return={medium_summary['avg_return']:.2f}, avg_cost={medium_summary['avg_cost']:.2f}")
    else:
        print(f"\n[2/4] Skipping medium collection (--skip-medium)")

    # --- Collect good (expert checkpoint) ---
    if not args.skip_good:
        print(f"\n[3/4] Collecting good (expert) data ...")
        n_good_needed = int(args.replay_total * good_ratio)
        n_good_collect = max(n_good_needed, 200)
        collect_cmd = [
            sys.executable,
            "offline_safe/dataset/build_dataset.py",
            "--scenario", args.scenario,
            "--mode", args.mode,
            "--voltage-barrier-type", args.voltage_barrier_type,
            "--episode-limit", str(args.episode_limit),
            "--episodes", str(n_good_collect),
            "--seed", str(args.seed),
            "--policy", "expert_checkpoint",
            "--alg", args.alg,
            "--checkpoint", args.checkpoint,
            "--save-dir", good_dir,
            "--start-day", str(args.start_day),
            "--day-step", str(args.day_step),
            "--hour", str(args.hour),
            "--quarter", str(args.quarter),
        ]
        if args.manual_reset:
            collect_cmd.append("--manual-reset")
        run_cmd(collect_cmd)
        good_summary = summarize_dataset_dir(good_dir, "good")
        save_json(os.path.join(good_dir, "summary.json"), good_summary)
        print(f"  good: {good_summary['num_episodes']} eps, avg_return={good_summary['avg_return']:.2f}, avg_cost={good_summary['avg_cost']:.2f}")
    else:
        print(f"\n[3/4] Skipping good collection (--skip-good)")

    # --- Mix ---
    if not args.skip_mix:
        print(f"\n[4/4] Mixing replay ({args.replay_total} total) ...")
        replay_summary = mix_replay(
            total_episodes=args.replay_total,
            poor_dir=poor_dir,
            medium_dir=medium_dir,
            good_dir=good_dir,
            replay_dir=replay_dir,
            ratios=ratios,
            seed=args.seed,
        )
        save_json(os.path.join(replay_dir, "summary.json"), replay_summary)
        print(f"  replay: {replay_summary['num_episodes']} eps, avg_return={replay_summary['avg_return']:.2f}, avg_cost={replay_summary['avg_cost']:.2f}")

    all_summary = {
        "alg": args.alg,
        "ratios": {"poor": args.poor_ratio, "medium": args.medium_ratio, "good": good_ratio},
        "replay_total": args.replay_total,
    }
    for split_name, d in [("poor", poor_dir), ("medium", medium_dir), ("good", good_dir), ("replay", replay_dir)]:
        sfp = os.path.join(d, "summary.json")
        if os.path.exists(sfp):
            with open(sfp, "r") as f:
                all_summary[split_name] = json.load(f)
    save_json(os.path.join(model_root, "all_summary.json"), all_summary)
    print(f"\nDone. All summary saved to {os.path.join(model_root, 'all_summary.json')}")


if __name__ == "__main__":
    main()
