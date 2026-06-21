import argparse
import json
import os
from typing import Dict, List
from typing import Dict, List, Optional
from offline_safe.eval.evaluate_baseline import evaluate_baseline
from baseline_checkpoints import SUPPORTED_BASELINE_MODELS, build_default_checkpoint_map


DEFAULT_MODELS = list(SUPPORTED_BASELINE_MODELS)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def aggregate_summaries(window_results: List[Dict]) -> Dict:
    if not window_results:
        return {}

    keys_mean = [
        "avg_return",
        "avg_cost",
        "avg_discounted_cost",
        "avg_v_out",
        "avg_destroy",
        "avg_q_loss",
        "avg_length",
        "cr",
        "pl",
        "v_dev",
        "max_v_drop_dev",
        "max_v_rise_dev",
        "no_destroy_rate",
        "worst_episode_cost",
        "worst_episode_v_out",
        "cvar95_cost",
    ]

    out = {
        "num_windows": len(window_results),
    }

    for k in keys_mean:
        vals = [x["summary"].get(k, 0.0) for x in window_results if x.get("summary") is not None]
        out[f"mean_{k}"] = float(sum(vals) / len(vals)) if vals else 0.0
        out[f"max_{k}"] = float(max(vals)) if vals else 0.0
        out[f"min_{k}"] = float(min(vals)) if vals else 0.0

    # 约束满足率单独处理：可能是 None
    csr_vals = [
        x["summary"].get("constraint_satisfaction_rate", None)
        for x in window_results
        if x.get("summary") is not None and x["summary"].get("constraint_satisfaction_rate", None) is not None
    ]
    out["mean_constraint_satisfaction_rate"] = (float(sum(csr_vals) / len(csr_vals)) if csr_vals else None)

    return out


def evaluate_one_window(
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
    cost_limit: Optional[float],
):
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
    return summary, records


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models across multiple time windows for cross-time generalization."
    )
    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--baseline-alias", type=str, default="0")
    parser.add_argument("--baseline-save-root", type=str, default="./")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)
    parser.add_argument("--cost-limit", type=float, default=5.6)

    # 跨时间段窗口
    parser.add_argument("--start-days", type=int, nargs="+", required=True)

    parser.add_argument("--results-dir", type=str, default="./results/baseline_generalization")
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
    all_results = {}

    for alg in args.models:
        if alg not in checkpoint_map:
            print(f"[skip] {alg}: no checkpoint entry configured")
            continue

        checkpoint_path = checkpoint_map[alg]
        if not os.path.exists(checkpoint_path):
            print(f"[skip] {alg}: checkpoint not found -> {checkpoint_path}")
            continue

        print(f"\n=== Cross-time evaluation for baseline: {alg} ===")

        per_window = []
        for sd in args.start_days:
            print(f"  -> start_day = {sd}")
            summary, records = evaluate_one_window(
                alg=alg,
                checkpoint_path=checkpoint_path,
                scenario=args.scenario,
                mode=args.mode,
                voltage_barrier_type=args.voltage_barrier_type,
                episode_limit=args.episode_limit,
                eval_episodes=args.eval_episodes,
                manual_reset=args.manual_reset,
                start_day=sd,
                day_step=args.day_step,
                hour=args.hour,
                quarter=args.quarter,
                cost_limit=args.cost_limit,
            )
            per_window.append({
                "start_day": sd,
                "summary": summary,
                "records": records,
            })

        aggregate = aggregate_summaries(per_window)
        aggregate["cost_limit"] = float(args.cost_limit) if args.cost_limit is not None else None

        result = {
            "alg": alg,
            "checkpoint_path": checkpoint_path,
            "scenario": args.scenario,
            "mode": args.mode,
            "voltage_barrier_type": args.voltage_barrier_type,
            "eval_episodes_per_window": args.eval_episodes,
            "start_days": args.start_days,
            "cost_limit": float(args.cost_limit) if args.cost_limit is not None else None,
            "window_results": per_window,
            "aggregate_summary": aggregate,
        }

        save_json(os.path.join(args.results_dir, f"{alg}.json"), result)
        all_results[alg] = result

        print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    save_json(os.path.join(args.results_dir, "all_baseline_generalization.json"), all_results)
    print(f"\nDone. Saved generalization results to: {args.results_dir}")


if __name__ == "__main__":
    main()