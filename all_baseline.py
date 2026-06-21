import argparse
import json
import os
from typing import Dict, List

from offline_safe.eval.evaluate_baseline import evaluate_baseline
from baseline_checkpoints import SUPPORTED_BASELINE_MODELS, build_default_checkpoint_map


DEFAULT_MODELS = list(SUPPORTED_BASELINE_MODELS)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def evaluate_one_model(
    alg: str,
    checkpoint_path: str,
    scenario: str,
    mode: str,
    voltage_barrier_type: str,
    episode_limit: int,
    eval_episodes: int,
    manual_reset: bool,
    start_day: int,
    day_step: int,
    hour: int,
    quarter: int,
    cost_limit: float | None,
) -> dict:
    summary, records = evaluate_baseline(
        model_path=checkpoint_path,
        alg=alg,
        scenario=scenario,
        mode=mode,
        voltage_barrier_type=voltage_barrier_type,
        episode_limit=episode_limit,
        eval_episodes=eval_episodes,
        manual_reset=manual_reset,
        start_day=start_day,
        day_step=day_step,
        hour=hour,
        quarter=quarter,
        cost_limit=cost_limit,
    )
    return {
        "alg": alg,
        "checkpoint_path": checkpoint_path,
        "cost_limit": cost_limit,
        "summary": summary,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate all baseline models with 15 safety/performance metrics.")
    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--baseline-alias", type=str, default="0")
    parser.add_argument("--baseline-save-root", type=str, default="./")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=730)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)
    parser.add_argument("--cost-limit", type=float, default=5.6)
    parser.add_argument("--results-dir", type=str, default="./results/baseline_all")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    checkpoint_map = build_default_checkpoint_map(
        save_root=args.baseline_save_root,
        scenario=args.scenario,
        mode=args.mode,
        voltage_barrier_type=args.voltage_barrier_type,
        alias=args.baseline_alias,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    aggregate: Dict[str, dict] = {}
    ranking: List[dict] = []

    for alg in args.models:
        if alg not in checkpoint_map:
            print(f"[skip] {alg}: no checkpoint entry configured.")
            continue
        checkpoint_path = checkpoint_map[alg]
        if not os.path.exists(checkpoint_path):
            print(f"[skip] {alg}: checkpoint not found -> {checkpoint_path}")
            continue

        print(f"\n=== Evaluating baseline: {alg} ===")
        result = evaluate_one_model(
            alg=alg,
            checkpoint_path=checkpoint_path,
            scenario=args.scenario,
            mode=args.mode,
            voltage_barrier_type=args.voltage_barrier_type,
            episode_limit=args.episode_limit,
            eval_episodes=args.eval_episodes,
            manual_reset=args.manual_reset,
            start_day=args.start_day,
            day_step=args.day_step,
            hour=args.hour,
            quarter=args.quarter,
            cost_limit=args.cost_limit,
        )
        save_json(os.path.join(args.results_dir, f"{alg}.json"), result)
        aggregate[alg] = result
        summary = result["summary"]
        ranking.append({
            "alg": alg,
            "avg_return": summary.get("avg_return", 0.0),
            "avg_cost": summary.get("avg_cost", 0.0),
            "avg_v_out": summary.get("avg_v_out", 0.0),
            "constraint_satisfaction_rate": summary.get("constraint_satisfaction_rate", 0.0),
            "cvar95_cost": summary.get("cvar95_cost", 0.0),
        })
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    ranking.sort(key=lambda x: (x["avg_cost"], x["avg_v_out"], -x["avg_return"]))
    save_json(os.path.join(args.results_dir, "all_baselines.json"), aggregate)
    save_json(os.path.join(args.results_dir, "ranking.json"), {"ranking": ranking})
    print(f"\nDone. Saved per-model results to: {args.results_dir}")


if __name__ == "__main__":
    main()