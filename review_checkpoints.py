import argparse
import json
import os

from offline_safe.dataset.replay_buffer import MAPDNTransitionDataset
from offline_safe.eval.evaluate_policy import evaluate_mabcq_lag


def make_key(summary):
    return (
        float(summary.get("worst_episode_cost", 1e18)),
        float(summary.get("cvar95_cost", 1e18)),
        -float(summary.get("avg_return", -1e18)),
        float(summary.get("avg_cost", 1e18)),
        float(summary.get("avg_v_out", 1e18)),
        -float(summary.get("cr", 0.0)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--epochs", nargs="+", required=True, help="e.g. 0080 0090 0100 0110 0120")
    parser.add_argument("--out-json", type=str, required=True)

    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=730)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--actor-model", type=str, default="mlp")
    parser.add_argument("--critic-model", type=str, default="maac")
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--critic-attend-heads", type=int, default=4)

    args = parser.parse_args()

    dataset = MAPDNTransitionDataset(
        args.data_roots,
        normalize_obs=True,
        normalize_state=True,
        cache_in_memory=True
    )

    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.json")

    action_low = -1.0
    action_high = 1.0
    for root in args.data_roots:
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

    all_results = []
    best = None
    best_key = None

    for ep in args.epochs:
        model_path = os.path.join(args.ckpt_dir, f"model_epoch_{ep}.pt")
        if not os.path.exists(model_path):
            print(f"Skip missing checkpoint: {model_path}")
            continue

        summary, records = evaluate_mabcq_lag(
            model_path=model_path,
            obs_dim=dataset.obs_dim,
            state_dim=dataset.state_dim,
            act_dim=dataset.act_dim,
            n_agents=dataset.n_agents,
            action_low=action_low,
            action_high=action_high,
            scenario=args.scenario,
            mode=args.mode,
            voltage_barrier_type=args.voltage_barrier_type,
            episode_limit=args.episode_limit,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            manual_reset=args.manual_reset,
            start_day=args.start_day,
            day_step=args.day_step,
            hour=args.hour,
            quarter=args.quarter,
            cost_limit=0.0,
            device=args.device,
            dataset_stats_path=stats_path,
            actor_model_name=args.actor_model,
            critic_model_name=args.critic_model,
            actor_hidden_dims=tuple(args.actor_hidden_dims),
            critic_hidden_dims=tuple(args.critic_hidden_dims),
            critic_attend_heads=args.critic_attend_heads,
        )

        item = {
            "epoch": ep,
            "model_path": model_path,
            "summary": summary,
            "records": records,
        }
        all_results.append(item)

        key = make_key(summary)
        if best_key is None or key < best_key:
            best_key = key
            best = item

    payload = {
        "eval_episodes": args.eval_episodes,
        "results": all_results,
        "best": best,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved review result to: {args.out_json}")
    if best is not None:
        print(f"Best checkpoint after review: epoch {best['epoch']}")


if __name__ == "__main__":
    main()