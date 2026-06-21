import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

from offline_safe.eval.judgement_relaxed import compare_with_relaxed_judgement

DEFAULT_MODELS = [
    "maddpg", "sqddpg", "iac", "iddpg", "coma",
    "maac", "matd3", "ippo", "mappo", "facmaddpg"
]


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def print_banner(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def run_cmd(cmd, stage_name: str = ""):
    if stage_name:
        print(f"[stage] {stage_name}")
    print("[cmd]", " ".join(cmd))
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)
    dt = time.time() - t0
    if stage_name:
        print(f"[done] {stage_name} | elapsed = {format_seconds(dt)}")
    return dt


def build_final_comparison(baseline_summary: dict, offline_summary: dict) -> dict:
    return compare_with_relaxed_judgement(baseline_summary, offline_summary)


def main():
    parser = argparse.ArgumentParser(
        description="Train one MABCQRetainLag-v3 model with actor/critic split data input, then run multi-window split review and final evaluation."
    )
    parser.add_argument("--dataset-root", type=str, default="./offline_safe/data/offline_case33_all")
    parser.add_argument("--baseline-results-dir", type=str, default="./results/baseline_all")
    parser.add_argument("--target-baseline-json", type=str, default="")
    parser.add_argument("--save-root", type=str, default="./results/offline_retain_v3")
    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=730)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--cost-limit", type=float, default=None,
                        help="Override cost limit (if None, estimate from dataset)")

    parser.add_argument("--actor-lr", type=float, default=3e-4,
                        help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=3e-4,
                        help="Critic learning rate")
    parser.add_argument("--vae-lr", type=float, default=3e-4,
                        help="VAE learning rate")
    parser.add_argument("--topk-checkpoints", type=int, default=5,
                        help="Number of top checkpoints to keep during training")

    parser.add_argument("--cost-limit-ratio", type=float, default=0.8)
    parser.add_argument("--cost-limit-quantile", type=float, default=0.5)
    parser.add_argument("--bc-coef", type=float, default=0.20)
    parser.add_argument("--bc-coef-end", type=float, default=0.05)
    parser.add_argument("--cql-alpha-reward", type=float, default=0.05)
    parser.add_argument("--cql-alpha-cost", type=float, default=0.10)
    parser.add_argument("--lag-lr", type=float, default=0.0002)
    parser.add_argument("--warmup-actor-steps", type=int, default=700)
    parser.add_argument("--lambda-warmup-steps", type=int, default=5000)

    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-epochs", type=int, default=4)
    parser.add_argument("--stop-on-no-improve", action="store_true")
    parser.add_argument("--freeze-actor-after-patience", action="store_true")
    parser.add_argument("--actor-lr-patience", type=int, default=1)
    parser.add_argument("--actor-lr-decay", type=float, default=0.5)
    parser.add_argument("--actor-lr-min", type=float, default=1e-5)
    parser.add_argument("--use-actor-lr-plateau", action="store_true")

    parser.add_argument("--val-start-days", type=int, nargs="+", default=[730, 760, 790])
    parser.add_argument("--val-eval-episodes", type=int, default=10)
    parser.add_argument("--test-start-days", type=int, nargs="+", default=[820, 850])
    parser.add_argument("--test-eval-episodes", type=int, default=10)

    parser.add_argument("--actor-cost-quantile", type=float, default=0.60)
    parser.add_argument("--actor-disc-cost-quantile", type=float, default=0.60)
    parser.add_argument("--actor-return-quantile", type=float, default=0.30)
    parser.add_argument("--actor-min-keep-ratio", type=float, default=0.35)
    parser.add_argument("--actor-good-weight", type=float, default=1.0)
    parser.add_argument("--actor-medium-weight", type=float, default=0.35)
    parser.add_argument("--actor-bad-weight", type=float, default=0.0)
    parser.add_argument("--state-budget-quantile", type=float, default=0.5,
                        help="Quantile for state-dependent cost budget")
    parser.add_argument("--unsafe-weight-coef", type=float, default=1.5,
                        help="Coefficient for unsafe weight")
    parser.add_argument("--safety-margin-coef", type=float, default=0.25,
                        help="Safety margin coefficient for actor scoring")
    parser.add_argument("--cost-weight-coef", type=float, default=1.5,
                        help="Weight coefficient for cost critic")
    parser.add_argument("--budget-mix-ratio", type=float, default=0.7,
                        help="Mixing ratio between state budget and global cost limit")
    parser.add_argument("--soft-return-drop-ratio", type=float, default=0.10,
                        help="Soft return drop threshold")
    parser.add_argument("--max-return-drop-ratio", type=float, default=0.15,
                        help="Hard return drop threshold")
    parser.add_argument("--use-actor-filter-for-vae", action="store_true")
    parser.add_argument("--use-actor-filter-for-actor", action="store_true")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable in-memory dataset caching")

    parser.add_argument("--review-actor-model", type=str, default="mlp")
    parser.add_argument("--review-critic-model", type=str, default="maac")
    parser.add_argument("--review-actor-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--review-critic-hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--review-critic-attend-heads", type=int, default=4)

    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    aggregate: Dict[str, dict] = {}
    ranking: List[dict] = []

    total_requested = len(args.models)
    completed = 0
    skipped = 0
    total_start_time = time.time()

    print_banner("OFFLINE RETAIN V3 MULTI-WINDOW PIPELINE START")
    print(f"[config] dataset_root = {args.dataset_root}")
    print(f"[config] baseline_results_dir = {args.baseline_results_dir}")
    print(f"[config] save_root = {args.save_root}")
    print(f"[config] scenario = {args.scenario}")
    print(f"[config] mode = {args.mode}")
    print(f"[config] voltage_barrier_type = {args.voltage_barrier_type}")
    print(f"[config] models = {args.models}")
    print(f"[config] val_start_days = {args.val_start_days}")
    print(f"[config] test_start_days = {args.test_start_days}")
    print(f"[config] device = {args.device}")

    for idx, alg in enumerate(args.models, start=1):
        model_start_time = time.time()

        print_banner(f"[overall progress] model {idx}/{total_requested} -> {alg}")

        dataset_dir = os.path.join(args.dataset_root, alg, "replay")
        if not os.path.isdir(dataset_dir):
            print(f"[skip] {alg}: dataset dir not found -> {dataset_dir}")
            skipped += 1
            continue

        baseline_result_path = args.target_baseline_json if args.target_baseline_json else os.path.join(args.baseline_results_dir, f"{alg}.json")
        if not os.path.exists(baseline_result_path):
            print(f"[skip] {alg}: baseline result not found -> {baseline_result_path}")
            skipped += 1
            continue

        save_dir = os.path.join(args.save_root, alg)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[model] source_baseline = {alg}")
        print(f"[model] dataset_dir = {dataset_dir}")
        print(f"[model] baseline_result_path = {baseline_result_path}")
        print(f"[model] save_dir = {save_dir}")

        train_cmd = [
            sys.executable,
            "offline_safe/trainers/train_offline_retain_v3.py",
            "--data-roots", dataset_dir,
            "--baseline-json", baseline_result_path,
            "--save-dir", save_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--scenario", args.scenario,
            "--mode", args.mode,
            "--voltage-barrier-type", args.voltage_barrier_type,
            "--episode-limit", str(args.episode_limit),
            "--seed", str(args.seed),
            "--eval-every", str(args.eval_every),
            "--eval-episodes", str(args.eval_episodes),
            "--start-day", str(args.start_day),
            "--day-step", str(args.day_step),
            "--hour", str(args.hour),
            "--quarter", str(args.quarter),
            "--cost-limit-ratio", str(args.cost_limit_ratio),
            "--cost-limit-quantile", str(args.cost_limit_quantile),
            "--bc-coef", str(args.bc_coef),
            "--bc-coef-end", str(args.bc_coef_end),
            "--cql-alpha-reward", str(args.cql_alpha_reward),
            "--cql-alpha-cost", str(args.cql_alpha_cost),
            "--lag-lr", str(args.lag_lr),
            "--warmup-actor-steps", str(args.warmup_actor_steps),
            "--lambda-warmup-steps", str(args.lambda_warmup_steps),
            "--actor-model", args.review_actor_model,
            "--critic-model", args.review_critic_model,
            "--actor-hidden-dims", *[str(x) for x in args.review_actor_hidden_dims],
            "--critic-hidden-dims", *[str(x) for x in args.review_critic_hidden_dims],
            "--critic-attend-heads", str(args.review_critic_attend_heads),
            "--early-stop-patience", str(args.early_stop_patience),
            "--early-stop-min-epochs", str(args.early_stop_min_epochs),
            "--actor-lr-patience", str(args.actor_lr_patience),
            "--actor-lr-decay", str(args.actor_lr_decay),
            "--actor-lr-min", str(args.actor_lr_min),
            "--do-eval",
            "--actor-lr", str(args.actor_lr),
            "--critic-lr", str(args.critic_lr),
            "--vae-lr", str(args.vae_lr),
            "--topk-checkpoints", str(args.topk_checkpoints),
            "--actor-cost-quantile", str(args.actor_cost_quantile),
            "--actor-disc-cost-quantile", str(args.actor_disc_cost_quantile),
            "--actor-return-quantile", str(args.actor_return_quantile),
            "--actor-min-keep-ratio", str(args.actor_min_keep_ratio),
            "--actor-good-weight", str(args.actor_good_weight),
            "--actor-medium-weight", str(args.actor_medium_weight),
            "--actor-bad-weight", str(args.actor_bad_weight),
            "--state-budget-quantile", str(args.state_budget_quantile),
            "--unsafe-weight-coef", str(args.unsafe_weight_coef),
            "--safety-margin-coef", str(args.safety_margin_coef),
            "--cost-weight-coef", str(args.cost_weight_coef),
            "--budget-mix-ratio", str(args.budget_mix_ratio),
        ]
        if args.manual_reset:
            train_cmd.append("--manual-reset")
        if args.use_actor_filter_for_vae:
            train_cmd.append("--use-actor-filter-for-vae")
        if args.use_actor_filter_for_actor:
            train_cmd.append("--use-actor-filter-for-actor")
        if args.stop_on_no_improve:
            train_cmd.append("--stop-on-no-improve")
        if args.freeze_actor_after_patience:
            train_cmd.append("--freeze-actor-after-patience")
        if args.use_actor_lr_plateau:
            train_cmd.append("--use-actor-lr-plateau")
        if args.no_cache:
            train_cmd.append("--no-cache")
        if args.cost_limit is not None:
            train_cmd.extend(["--cost-limit", str(args.cost_limit)])

        train_elapsed = run_cmd(train_cmd, stage_name=f"{alg} train_offline_retain_v1")

        review_json = os.path.join(save_dir, "final_split_review.json")
        review_cmd = [
            sys.executable,
            "review_checkpoints_split_v2.py",
            "--data-roots", dataset_dir,
            "--ckpt-dir", save_dir,
            "--out-json", review_json,
            "--min-epoch", "1",
            "--max-epoch", str(args.epochs),
            "--stride", "1",
            "--scenario", args.scenario,
            "--mode", args.mode,
            "--voltage-barrier-type", args.voltage_barrier_type,
            "--episode-limit", str(args.episode_limit),
            "--seed", str(args.seed),
            "--day-step", str(args.day_step),
            "--hour", str(args.hour),
            "--quarter", str(args.quarter),
            "--device", args.device,
            "--val-start-days", *[str(x) for x in args.val_start_days],
            "--val-eval-episodes", str(args.val_eval_episodes),
            "--test-start-days", *[str(x) for x in args.test_start_days],
            "--test-eval-episodes", str(args.test_eval_episodes),
            "--actor-model", args.review_actor_model,
            "--critic-model", args.review_critic_model,
            "--actor-hidden-dims", *[str(x) for x in args.review_actor_hidden_dims],
            "--critic-hidden-dims", *[str(x) for x in args.review_critic_hidden_dims],
            "--critic-attend-heads", str(args.review_critic_attend_heads),
            "--cost-limit-ratio", str(args.cost_limit_ratio),
            "--cost-limit-quantile", str(args.cost_limit_quantile),
            "--baseline-json", baseline_result_path,
            "--soft-return-drop-ratio", str(args.soft_return_drop_ratio),
            "--max-return-drop-ratio", str(args.max_return_drop_ratio),
        ]
        if args.manual_reset:
            review_cmd.append("--manual-reset")

        review_elapsed = run_cmd(review_cmd, stage_name=f"{alg} review_checkpoints_split_v1")

        if not os.path.exists(review_json):
            raise FileNotFoundError(f"split review result not found: {review_json}")

        compare_start = time.time()
        with open(review_json, "r", encoding="utf-8") as f:
            review_payload = json.load(f)

        final_test_result = review_payload.get("final_test_result", None)
        if final_test_result is None:
            raise ValueError(f"final_test_result missing in: {review_json}")

        offline_summary = final_test_result["summary"]
        offline_records = final_test_result["records"]

        def convert_aggregate_summary(agg):
            mapping = {
                "mean_avg_return": "avg_return",
                "mean_avg_cost": "avg_cost",
                "mean_avg_discounted_cost": "avg_discounted_cost",
                "mean_avg_v_out": "avg_v_out",
                "mean_avg_destroy": "avg_destroy",
                "mean_avg_q_loss": "avg_q_loss",
                "mean_avg_length": "avg_length",
                "mean_cr": "cr",
                "mean_pl": "pl",
                "mean_v_dev": "v_dev",
                "mean_max_v_drop_dev": "max_v_drop_dev",
                "mean_max_v_rise_dev": "max_v_rise_dev",
                "mean_no_destroy_rate": "no_destroy_rate",
                "mean_worst_episode_cost": "worst_episode_cost",
                "mean_worst_episode_v_out": "worst_episode_v_out",
                "mean_cvar95_cost": "cvar95_cost",
                "mean_constraint_satisfaction_rate": "constraint_satisfaction_rate",
            }
            return {dst: agg[src] for src, dst in mapping.items() if src in agg}

        def load_any_baseline_summary(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("final_test_result") is not None:
                s = data["final_test_result"].get("summary")
                if s is not None:
                    return s

            if data.get("selected_best_on_validation") is not None:
                s = data["selected_best_on_validation"].get("summary")
                if s is not None:
                    return s

            if "summary" in data:
                return data["summary"]

            if "aggregate_summary" in data:
                return convert_aggregate_summary(data["aggregate_summary"])

            return data

        baseline_summary = load_any_baseline_summary(baseline_result_path)

        compare = build_final_comparison(
            baseline_summary=baseline_summary,
            offline_summary=offline_summary,
        )

        result = {
            "source_baseline": alg,
            "dataset_dir": dataset_dir,
            "train_save_dir": save_dir,
            "review_json": review_json,
            "selected_epoch_on_validation": final_test_result.get("epoch", ""),
            "model_path": final_test_result.get("model_path", ""),
            "baseline_summary": baseline_summary,
            "offline_summary": offline_summary,
            "offline_records": offline_records,
            "comparison": compare,
            "timing": {
                "train_seconds": train_elapsed,
                "review_seconds": review_elapsed,
            }
        }
        save_json(os.path.join(save_dir, "offline_eval.json"), result)
        aggregate[alg] = result

        ranking.append({
            "source_baseline": alg,
            "avg_return": offline_summary.get("avg_return", 0.0),
            "avg_cost": offline_summary.get("avg_cost", 0.0),
            "avg_v_out": offline_summary.get("avg_v_out", 0.0),
            "constraint_satisfaction_rate": offline_summary.get("constraint_satisfaction_rate", 0.0)
                if offline_summary.get("constraint_satisfaction_rate", None) is not None else 0.0,
            "cvar95_cost": offline_summary.get("cvar95_cost", 0.0),
            "overall_claim_supported": compare.get("overall_claim_supported", False),
        })

        compare_elapsed = time.time() - compare_start
        model_elapsed = time.time() - model_start_time
        completed += 1

        print(f"[done] {alg} comparison | elapsed = {format_seconds(compare_elapsed)}")
        print(f"[summary] {alg} offline_summary:")
        print(json.dumps(offline_summary, indent=2, ensure_ascii=False))
        print(f"[summary] {alg} comparison:")
        print(json.dumps(compare, indent=2, ensure_ascii=False))
        print(
            f"[model done] {alg} | train = {format_seconds(train_elapsed)} | "
            f"review = {format_seconds(review_elapsed)} | total = {format_seconds(model_elapsed)}"
        )
        print(
            f"[overall status] completed = {completed}/{total_requested}, "
            f"skipped = {skipped}, remaining = {total_requested - idx}"
        )

    ranking.sort(key=lambda x: (x["avg_cost"], x["avg_v_out"], -x["avg_return"]))
    save_json(os.path.join(args.save_root, "all_offline_results.json"), aggregate)
    save_json(os.path.join(args.save_root, "ranking.json"), {"ranking": ranking})

    total_elapsed = time.time() - total_start_time
    print_banner("OFFLINE RETAIN V3 MULTI-WINDOW PIPELINE FINISHED")
    print(f"[final] total_requested = {total_requested}")
    print(f"[final] completed = {completed}")
    print(f"[final] skipped = {skipped}")
    print(f"[final] total_elapsed = {format_seconds(total_elapsed)}")
    print(f"[final] results saved to = {args.save_root}")


if __name__ == "__main__":
    main()