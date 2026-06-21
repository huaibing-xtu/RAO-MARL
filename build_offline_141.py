import argparse
import json
import os
import subprocess
import sys
from typing import Dict

from models.model_registry import Model

DEFAULT_MODELS = list(Model.keys())

DEFAULT_CHECKPOINTS = {
    "maddpg": "./model_save/var_voltage_control-case141_3min_final-distributed-maddpg-l1-1/model.pt",
    "sqddpg": "./model_save/var_voltage_control-case141_3min_final-distributed-sqddpg-l1-1/model.pt",
    "iac": "./model_save/var_voltage_control-case141_3min_final-distributed-iac-l1-1/model.pt",
    "iddpg": "./model_save/var_voltage_control-case141_3min_final-distributed-iddpg-l1-1/model.pt",
    "coma": "./model_save/var_voltage_control-case141_3min_final-distributed-coma-l1-1/model.pt",
    "maac": "./model_save/var_voltage_control-case141_3min_final-distributed-maac-l1-1/model.pt",
    "matd3": "./model_save/var_voltage_control-case141_3min_final-distributed-matd3-l1-1/model.pt",
    "ippo": "./model_save/var_voltage_control-case141_3min_final-distributed-ippo-l1-1/model.pt",
    "mappo": "./model_save/var_voltage_control-case141_3min_final-distributed-mappo-l1-1/model.pt",
    "facmaddpg": "./model_save/var_voltage_control-case141_3min_final-distributed-facmaddpg-l1-1/model.pt",
}


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Build one expert offline dataset for each baseline checkpoint (case141).")
    parser.add_argument("--scenario", type=str, default="case141_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=0)
    parser.add_argument("--quarter", type=int, default=0)
    parser.add_argument("--dataset-root", type=str, default="./offline_safe/data/offline_case141_all")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    os.makedirs(args.dataset_root, exist_ok=True)
    manifest: Dict[str, dict] = {}

    for alg in args.models:
        checkpoint_path = DEFAULT_CHECKPOINTS.get(alg, "")
        if not checkpoint_path:
            print(f"[skip] {alg}: no checkpoint entry configured.")
            continue
        if not os.path.exists(checkpoint_path):
            print(f"[skip] {alg}: checkpoint not found -> {checkpoint_path}")
            continue

        save_dir = os.path.join(args.dataset_root, alg, "replay")
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n=== Collect offline dataset from baseline: {alg} ===")

        cmd = [
            sys.executable,
            "offline_safe/dataset/build_dataset.py",
            "--scenario", args.scenario,
            "--mode", args.mode,
            "--voltage-barrier-type", args.voltage_barrier_type,
            "--episode-limit", str(args.episode_limit),
            "--episodes", str(args.episodes),
            "--seed", str(args.seed),
            "--policy", "expert_checkpoint",
            "--alg", alg,
            "--checkpoint", checkpoint_path,
            "--save-dir", save_dir,
            "--start-day", str(args.start_day),
            "--day-step", str(args.day_step),
            "--hour", str(args.hour),
            "--quarter", str(args.quarter),
        ]
        if args.manual_reset:
            cmd.append("--manual-reset")
        run_cmd(cmd)

        meta_path = os.path.join(save_dir, "meta.json")
        manifest[alg] = {
            "alg": alg,
            "checkpoint_path": checkpoint_path,
            "dataset_dir": save_dir,
            "meta_path": meta_path,
        }

    save_json(os.path.join(args.dataset_root, "dataset_manifest.json"), manifest)
    print(f"\nDone. Saved dataset manifest to: {os.path.join(args.dataset_root, 'dataset_manifest.json')}")


if __name__ == "__main__":
    main()
