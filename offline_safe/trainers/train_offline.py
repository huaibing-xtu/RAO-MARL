import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from offline_safe.dataset.replay_buffer import MAPDNTransitionDataset
from offline_safe.algos.ma_bcq_lag import MABCQLag
from offline_safe.eval.evaluate_policy import evaluate_mabcq_lag


def aggregate_logs(logs: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    if len(logs) == 0:
        return out
    for key in logs[0].keys():
        out[key] = float(np.mean([x[key] for x in logs]))
    return out

def estimate_cost_limit(
    dataset: MAPDNTransitionDataset,
    gamma: float = 0.99,
    ratio: float = 0.8,
    quantile: float = 0.5,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", nargs="+", required=True, help="例如: ./offline_safe/data/offline_case33/replay")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--action-low", type=float, default=None)
    parser.add_argument("--action-high", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=10)

    # model selection
    parser.add_argument("--actor-model", type=str, default="mlp", choices=["mlp", "residual"], help="共享 actor 模型类型")
    parser.add_argument("--critic-model", type=str, default="mlp", choices=["mlp", "central", "maac"], help="中心化 critic 模型类型")
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[256, 256], help="actor / behavior MLP hidden dims")
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[512, 512], help="critic hidden dims")
    parser.add_argument("--critic-attend-heads", type=int, default=4, help="仅 critic-model=maac 时使用")

    # eval
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=3)
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

    parser.add_argument("--cost-limit-quantile", type=float, default=0.5)
    parser.add_argument("--bc-coef", type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = MAPDNTransitionDataset(args.data_roots, normalize_obs=True, normalize_state=True, cache_in_memory=True)
    dataset.save_stats(os.path.join(args.save_dir, "dataset_stats.json"))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # 从原始 dataset meta 里估计动作边界更稳；这里用数据中的动作极值也能先跑通
    action_low = -1.0 if args.action_low is None else float(args.action_low)
    action_high = 1.0 if args.action_high is None else float(args.action_high)
    # 尝试从 meta.json 读取边界
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
            quantile=args.cost_limit_quantile,
        )
        print(f"[info] auto cost_limit = {cost_limit:.6f}")
    else:
        cost_limit = float(args.cost_limit)
        print(f"[info] manual cost_limit = {cost_limit:.6f}")

    algo = MABCQLag(
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
        actor_model_name=args.actor_model,
        critic_model_name=args.critic_model,
        actor_hidden_dims=tuple(args.actor_hidden_dims),
        critic_hidden_dims=tuple(args.critic_hidden_dims),
        critic_attend_heads=args.critic_attend_heads,
        obs_mean=dataset.obs_mean,
        obs_std=dataset.obs_std,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
    )

    train_history: List[Dict] = []
    best_score = None

    for epoch in range(1, args.epochs + 1):
        logs = []
        for batch in loader:
            out = algo.update(batch)
            logs.append(out)

        epoch_log = aggregate_logs(logs)
        epoch_log["epoch"] = epoch
        train_history.append(epoch_log)

        print(
            f"[epoch {epoch:04d}] "
            f"vae={epoch_log.get('vae_loss', 0.0):.4f} | "
            f"qr={epoch_log.get('qr_loss', 0.0):.4f} | "
            f"qc={epoch_log.get('qc_loss', 0.0):.4f} | "
            f"actor={epoch_log.get('actor_loss', 0.0):.4f} | "
            f"lambda={epoch_log.get('lambda', 0.0):.4f} | "
            f"qr_pi={epoch_log.get('qr_pi', 0.0):.4f} | "
            f"qc_pi={epoch_log.get('qc_pi', 0.0):.4f}"
        )

        if args.do_eval and (epoch % args.eval_every == 0 or epoch == args.epochs):
            latest_path = os.path.join(args.save_dir, f"model_epoch_{epoch:04d}.pt")

            algo.save(latest_path)

            summary, _ = evaluate_mabcq_lag(
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
                dataset_stats_path=os.path.join(args.save_dir, "dataset_stats.json"),
            )
            print(f"[eval epoch {epoch:04d}] {json.dumps(summary, ensure_ascii=False)}")
            epoch_log["eval"] = summary

            score_tuple = (
                summary["avg_cost"],
                summary["avg_v_out"],
                -summary["avg_return"],
            )

            if best_score is None or score_tuple < best_score:
                best_score = score_tuple
                best_path = os.path.join(args.save_dir, "model_best.pt")
                algo.save(best_path)
                with open(os.path.join(args.save_dir, "best_eval.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(os.path.join(args.save_dir, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump(train_history, f, indent=2, ensure_ascii=False)

    print(f"Training finished. Save dir: {args.save_dir}")


if __name__ == "__main__":
    main()
