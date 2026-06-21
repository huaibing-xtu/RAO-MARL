import argparse
import json
import os
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader

from offline_safe.dataset.replay_buffer import MAPDNTransitionDataset
from offline_safe.eval.evaluate_policy import evaluate_mabcq_lag
from offline_safe.eval.judgement_relaxed import compare_with_relaxed_judgement
from offline_safe.algos.ma_bcq_retain_lag import MABCQRetainLag


def aggregate_logs(logs: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    if not logs:
        return out
    for key in logs[0]:
        out[key] = float(np.mean([x[key] for x in logs]))
    return out


def estimate_cost_limit(
    dataset: MAPDNTransitionDataset,
    gamma: float = 0.99,
    ratio: float = 0.8,
    quantile: float = 0.5
) -> float:
    discounted_episode_costs = []
    for epi_data in dataset.data:
        costs = epi_data["costs"].reshape(-1)
        disc = 0.0
        for t, c in enumerate(costs):
            disc += (gamma ** t) * float(c)
        discounted_episode_costs.append(disc)
    base = float(np.quantile(discounted_episode_costs, quantile))
    return ratio * base


def load_baseline_summary(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" in data:
        return data["summary"]
    return data


def make_best_model_key(summary: Dict[str, float]):
    """
    Smaller tuple is better.

    Priority:
    1) worst_episode_cost        lower is better
    2) cvar95_cost              lower is better
    3) avg_return               higher is better
    4) avg_cost                 lower is better
    5) avg_v_out                lower is better
    6) cr                       higher is better
    """
    worst_episode_cost = float(summary.get("worst_episode_cost", 1e18))
    cvar95_cost = float(summary.get("cvar95_cost", 1e18))
    avg_return = float(summary.get("avg_return", -1e18))
    avg_cost = float(summary.get("avg_cost", 1e18))
    avg_v_out = float(summary.get("avg_v_out", 1e18))
    cr = float(summary.get("cr", 0.0))

    return (
        worst_episode_cost,
        cvar95_cost,
        -avg_return,
        avg_cost,
        avg_v_out,
        -cr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--baseline-json", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--phi", type=float, default=0.05)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--vae-lr", type=float, default=3e-4)
    parser.add_argument("--lag-lr", type=float, default=1e-4)
    parser.add_argument("--cost-limit", type=float, default=None)
    parser.add_argument("--cost-limit-ratio", type=float, default=0.8)
    parser.add_argument("--cost-limit-quantile", type=float, default=0.5)
    parser.add_argument("--action-low", type=float, default=None)
    parser.add_argument("--action-high", type=float, default=None)
    parser.add_argument("--bc-coef", type=float, default=0.25)
    parser.add_argument("--bc-coef-end", type=float, default=0.05)
    parser.add_argument("--action-l2-coef", type=float, default=1e-4)
    parser.add_argument("--cql-alpha-reward", type=float, default=0.05)
    parser.add_argument("--cql-alpha-cost", type=float, default=0.05)
    parser.add_argument("--warmup-actor-steps", type=int, default=1000)
    parser.add_argument("--actor-model", type=str, default="mlp", choices=["mlp", "residual"])
    parser.add_argument("--critic-model", type=str, default="maac", choices=["mlp", "central", "maac"])
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--critic-attend-heads", type=int, default=4)
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=730)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = MAPDNTransitionDataset(
        args.data_roots,
        normalize_obs=True,
        normalize_state=True,
        cache_in_memory=True
    )
    stats_path = os.path.join(args.save_dir, "dataset_stats.json")
    dataset.save_stats(stats_path)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    action_low = -1.0 if args.action_low is None else float(args.action_low)
    action_high = 1.0 if args.action_high is None else float(args.action_high)
    for root in args.data_roots:
        meta_path = os.path.join(root, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            try:
                action_low = float(np.min(meta["action_low"]))
                action_high = float(np.max(meta["action_high"]))
                break
            except Exception:
                pass

    if args.cost_limit is None:
        cost_limit = estimate_cost_limit(
            dataset,
            gamma=args.gamma,
            ratio=args.cost_limit_ratio,
            quantile=args.cost_limit_quantile
        )
    else:
        cost_limit = float(args.cost_limit)

    algo = MABCQRetainLag(
        obs_dim=dataset.obs_dim,
        state_dim=dataset.state_dim,
        act_dim=dataset.act_dim,
        n_agents=dataset.n_agents,
        action_low=action_low,
        action_high=action_high,
        device=args.device,
        gamma=args.gamma,
        tau=args.tau,
        phi=args.phi,
        latent_dim=args.latent_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        vae_lr=args.vae_lr,
        lag_lr=args.lag_lr,
        cost_limit=cost_limit,
        bc_coef=args.bc_coef,
        bc_coef_end=args.bc_coef_end,
        action_l2_coef=args.action_l2_coef,
        cql_alpha_reward=args.cql_alpha_reward,
        cql_alpha_cost=args.cql_alpha_cost,
        warmup_actor_steps=args.warmup_actor_steps,
        actor_model_name=args.actor_model,
        critic_model_name=args.critic_model,
        actor_hidden_dims=tuple(args.actor_hidden_dims),
        critic_hidden_dims=tuple(args.critic_hidden_dims),
        critic_attend_heads=args.critic_attend_heads,
    )

    train_history = []
    baseline_summary = load_baseline_summary(args.baseline_json) if args.baseline_json else None
    best_score = None

    for epoch in range(1, args.epochs + 1):
        logs = []
        for batch in loader:
            logs.append(algo.update(batch))
        epoch_log = aggregate_logs(logs)
        epoch_log["epoch"] = epoch
        train_history.append(epoch_log)

        if args.do_eval and (epoch % args.eval_every == 0 or epoch == args.epochs):
            latest_path = os.path.join(args.save_dir, f"model_epoch_{epoch:04d}.pt")
            algo.save(latest_path)

            summary, records = evaluate_mabcq_lag(
                model_path=latest_path,
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
                cost_limit=cost_limit,
                device=args.device,
                dataset_stats_path=stats_path,
                actor_model_name=args.actor_model,
                critic_model_name=args.critic_model,
                actor_hidden_dims=tuple(args.actor_hidden_dims),
                critic_hidden_dims=tuple(args.critic_hidden_dims),
                critic_attend_heads=args.critic_attend_heads,
            )

            epoch_log["eval"] = summary
            comparison = None
            if baseline_summary is not None:
                try:
                    comparison = compare_with_relaxed_judgement(baseline_summary, summary)
                except Exception as e:
                    comparison = {
                        "comparison_error": str(e)
                    }

            if comparison is not None:
                epoch_log["comparison"] = comparison

            candidate_key = make_best_model_key(summary)

            if best_score is None or candidate_key < best_score:
                best_score = candidate_key

                best_path = os.path.join(args.save_dir, "model_best.pt")
                algo.save(best_path)

                payload = {
                    "summary": summary,
                    "records": records,
                    "best_model_key": {
                        "worst_episode_cost": float(summary.get("worst_episode_cost", 1e18)),
                        "cvar95_cost": float(summary.get("cvar95_cost", 1e18)),
                        "avg_return": float(summary.get("avg_return", -1e18)),
                        "avg_cost": float(summary.get("avg_cost", 1e18)),
                        "avg_v_out": float(summary.get("avg_v_out", 1e18)),
                        "cr": float(summary.get("cr", 0.0)),
                    },
                    "epoch": epoch,
                }
                if comparison is not None:
                    payload["comparison"] = comparison

                with open(os.path.join(args.save_dir, "best_eval.json"), "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(os.path.join(args.save_dir, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump(train_history, f, indent=2, ensure_ascii=False)

    print(f"Training finished. Save dir: {args.save_dir}")


if __name__ == "__main__":
    main()