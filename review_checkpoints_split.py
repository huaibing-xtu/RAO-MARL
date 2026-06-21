import argparse
import json
import os
import glob

from offline_safe.dataset.replay_buffer import MAPDNTransitionDataset
from offline_safe.eval.evaluate_policy import evaluate_mabcq_lag


def make_key(summary):
    # 小者更优
    return (
        float(summary.get("worst_episode_cost", 1e18)),
        float(summary.get("cvar95_cost", 1e18)),
        -float(summary.get("avg_return", -1e18)),
        float(summary.get("avg_cost", 1e18)),
        float(summary.get("avg_v_out", 1e18)),
        -float(summary.get("cr", 0.0)),
    )


def discover_epochs(ckpt_dir, min_epoch, max_epoch, stride):
    epochs = []
    for p in glob.glob(os.path.join(ckpt_dir, "model_epoch_*.pt")):
        name = os.path.basename(p)
        ep = name.replace("model_epoch_", "").replace(".pt", "")
        try:
            ep_int = int(ep)
        except Exception:
            continue
        if ep_int < min_epoch or ep_int > max_epoch:
            continue
        if (ep_int - min_epoch) % stride != 0:
            continue
        epochs.append(f"{ep_int:04d}")
    return sorted(list(set(epochs)))


def load_dataset_and_action_space(data_roots):
    dataset = MAPDNTransitionDataset(
        data_roots,
        normalize_obs=True,
        normalize_state=True,
        cache_in_memory=True
    )

    action_low = -1.0
    action_high = 1.0
    for root in data_roots:
        meta_path = os.path.join(root, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            try:
                action_low = float(min(meta["action_low"]))
                action_high = float(max(meta["action_high"]))
                break
            except Exception:
                pass

    return dataset, action_low, action_high


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def aggregate_window_summaries(window_results):
    if not window_results:
        return None

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

    summary = {
        "eval_windows": len(window_results),
    }

    for k in keys_mean:
        vals = [w["summary"].get(k, 0.0) for w in window_results if w.get("summary") is not None]
        summary[k] = float(sum(vals) / len(vals)) if vals else 0.0

    csr_vals = [
        w["summary"].get("constraint_satisfaction_rate", None)
        for w in window_results
        if w.get("summary") is not None
    ]
    summary["constraint_satisfaction_rate"] = mean_or_none(csr_vals)
    return summary


def eval_one_model_one_window(
    model_path,
    dataset,
    action_low,
    action_high,
    stats_path,
    scenario,
    mode,
    voltage_barrier_type,
    episode_limit,
    seed,
    eval_episodes,
    manual_reset,
    start_day,
    day_step,
    hour,
    quarter,
    device,
    actor_model,
    critic_model,
    actor_hidden_dims,
    critic_hidden_dims,
    critic_attend_heads,
):
    summary, records = evaluate_mabcq_lag(
        model_path=model_path,
        obs_dim=dataset.obs_dim,
        state_dim=dataset.state_dim,
        act_dim=dataset.act_dim,
        n_agents=dataset.n_agents,
        action_low=action_low,
        action_high=action_high,
        scenario=scenario,
        mode=mode,
        voltage_barrier_type=voltage_barrier_type,
        episode_limit=episode_limit,
        seed=seed,
        eval_episodes=eval_episodes,
        manual_reset=manual_reset,
        start_day=start_day,
        day_step=day_step,
        hour=hour,
        quarter=quarter,
        cost_limit=0.0,
        device=device,
        dataset_stats_path=stats_path,
        actor_model_name=actor_model,
        critic_model_name=critic_model,
        actor_hidden_dims=tuple(actor_hidden_dims),
        critic_hidden_dims=tuple(critic_hidden_dims),
        critic_attend_heads=critic_attend_heads,
    )
    return summary, records


def eval_one_model_multi_windows(
    model_path,
    dataset,
    action_low,
    action_high,
    stats_path,
    scenario,
    mode,
    voltage_barrier_type,
    episode_limit,
    seed,
    eval_episodes,
    manual_reset,
    start_days,
    day_step,
    hour,
    quarter,
    device,
    actor_model,
    critic_model,
    actor_hidden_dims,
    critic_hidden_dims,
    critic_attend_heads,
):
    window_results = []
    merged_records = []

    for sd in start_days:
        summary, records = eval_one_model_one_window(
            model_path=model_path,
            dataset=dataset,
            action_low=action_low,
            action_high=action_high,
            stats_path=stats_path,
            scenario=scenario,
            mode=mode,
            voltage_barrier_type=voltage_barrier_type,
            episode_limit=episode_limit,
            seed=seed,
            eval_episodes=eval_episodes,
            manual_reset=manual_reset,
            start_day=sd,
            day_step=day_step,
            hour=hour,
            quarter=quarter,
            device=device,
            actor_model=actor_model,
            critic_model=critic_model,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            critic_attend_heads=critic_attend_heads,
        )
        window_results.append({
            "start_day": sd,
            "summary": summary,
            "records": records,
        })
        merged_records.extend(records)

    agg_summary = aggregate_window_summaries(window_results)
    return agg_summary, merged_records, window_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)

    parser.add_argument("--min-epoch", type=int, default=20)
    parser.add_argument("--max-epoch", type=int, default=80)
    parser.add_argument("--stride", type=int, default=5)

    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")

    # validation 多窗口
    parser.add_argument("--val-start-days", type=int, nargs="+", default=[730])
    parser.add_argument("--val-eval-episodes", type=int, default=20)

    # test 多窗口
    parser.add_argument("--test-start-days", type=int, nargs="+", default=[760])
    parser.add_argument("--test-eval-episodes", type=int, default=20)

    parser.add_argument("--actor-model", type=str, default="mlp")
    parser.add_argument("--critic-model", type=str, default="maac")
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--critic-attend-heads", type=int, default=4)

    args = parser.parse_args()

    dataset, action_low, action_high = load_dataset_and_action_space(args.data_roots)
    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.json")

    epochs = discover_epochs(args.ckpt_dir, args.min_epoch, args.max_epoch, args.stride)

    val_results = []
    best = None
    best_key = None

    # 1) validation review on multiple windows
    for ep in epochs:
        model_path = os.path.join(args.ckpt_dir, f"model_epoch_{ep}.pt")
        if not os.path.exists(model_path):
            continue

        summary, records, window_results = eval_one_model_multi_windows(
            model_path=model_path,
            dataset=dataset,
            action_low=action_low,
            action_high=action_high,
            stats_path=stats_path,
            scenario=args.scenario,
            mode=args.mode,
            voltage_barrier_type=args.voltage_barrier_type,
            episode_limit=args.episode_limit,
            seed=args.seed,
            eval_episodes=args.val_eval_episodes,
            manual_reset=args.manual_reset,
            start_days=args.val_start_days,
            day_step=args.day_step,
            hour=args.hour,
            quarter=args.quarter,
            device=args.device,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            actor_hidden_dims=args.actor_hidden_dims,
            critic_hidden_dims=args.critic_hidden_dims,
            critic_attend_heads=args.critic_attend_heads,
        )

        item = {
            "epoch": ep,
            "model_path": model_path,
            "summary": summary,
            "records": records,
            "window_results": window_results,
        }
        val_results.append(item)

        key = make_key(summary)
        if best_key is None or key < best_key:
            best_key = key
            best = item

    # 2) final test on held-out multiple windows
    test_result = None
    if best is not None:
        test_summary, test_records, test_window_results = eval_one_model_multi_windows(
            model_path=best["model_path"],
            dataset=dataset,
            action_low=action_low,
            action_high=action_high,
            stats_path=stats_path,
            scenario=args.scenario,
            mode=args.mode,
            voltage_barrier_type=args.voltage_barrier_type,
            episode_limit=args.episode_limit,
            seed=args.seed,
            eval_episodes=args.test_eval_episodes,
            manual_reset=args.manual_reset,
            start_days=args.test_start_days,
            day_step=args.day_step,
            hour=args.hour,
            quarter=args.quarter,
            device=args.device,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            actor_hidden_dims=args.actor_hidden_dims,
            critic_hidden_dims=args.critic_hidden_dims,
            critic_attend_heads=args.critic_attend_heads,
        )
        test_result = {
            "epoch": best["epoch"],
            "model_path": best["model_path"],
            "summary": test_summary,
            "records": test_records,
            "window_results": test_window_results,
        }

    payload = {
        "validation_eval_episodes_per_window": args.val_eval_episodes,
        "test_eval_episodes_per_window": args.test_eval_episodes,
        "val_start_days": args.val_start_days,
        "test_start_days": args.test_start_days,
        "validation_results": val_results,
        "selected_best_on_validation": best,
        "final_test_result": test_result,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved split review to: {args.out_json}")
    if best is not None:
        print(f"Selected best epoch on validation: {best['epoch']}")
        print(f"Final test completed for epoch: {best['epoch']}")


if __name__ == "__main__":
    main()