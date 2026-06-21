import argparse
import json
import os
from typing import Dict, List

import numpy as np

from offline_safe.dataset.common import build_env
from offline_safe.algos.ma_bcq_lag import MABCQLag
from offline_safe.eval.safety_metrics import EpisodeMetricTracker, summarize_records

def _normalize_obs(obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    return (obs - obs_mean.squeeze(0)) / (obs_std.squeeze(0) + 1e-8)


def _normalize_state(state: np.ndarray, state_mean: np.ndarray, state_std: np.ndarray) -> np.ndarray:
    return (state - state_mean.squeeze(0)) / (state_std.squeeze(0) + 1e-8)

def evaluate_mabcq_lag(
    model_path: str,
    obs_dim: int,
    state_dim: int,
    act_dim: int,
    n_agents: int,
    action_low: float,
    action_high: float,
    scenario: str,
    mode: str,
    voltage_barrier_type: str,
    episode_limit: int,
    seed: int,
    eval_episodes: int,
    manual_reset: bool,
    start_day: int,
    day_step: int,
    hour: int,
    quarter: int,
    cost_limit: float = 5.0,
    device: str = "gpu",
    dataset_stats_path: str = "",
    actor_model_name: str = "mlp",
    critic_model_name: str = "mlp",
    actor_hidden_dims=(256, 256),
    critic_hidden_dims=(512, 512),
    critic_attend_heads: int = 4,
):
    env, _ = build_env(
        env_name="var_voltage_control",
        scenario=scenario,
        mode=mode,
        voltage_barrier_type=voltage_barrier_type,
        episode_limit=episode_limit,
        seed=seed,
    )

    obs_mean = obs_std = state_mean = state_std = None
    if dataset_stats_path and os.path.exists(dataset_stats_path):
        with open(dataset_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)
        obs_std = np.asarray(stats["obs_std"], dtype=np.float32)
        state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
        state_std = np.asarray(stats["state_std"], dtype=np.float32)

    algo = MABCQLag(
        obs_dim=obs_dim,
        state_dim=state_dim,
        act_dim=act_dim,
        n_agents=n_agents,
        action_low=action_low,
        action_high=action_high,
        cost_limit=cost_limit,
        device=device,
        actor_model_name=actor_model_name,
        critic_model_name=critic_model_name,
        actor_hidden_dims=tuple(actor_hidden_dims),
        critic_hidden_dims=tuple(critic_hidden_dims),
        critic_attend_heads=critic_attend_heads,
        obs_mean=obs_mean,
        obs_std=obs_std,
        state_mean=state_mean,
        state_std=state_std,
    )
    algo.load(model_path)

    records: List[Dict] = []
    for ep in range(eval_episodes):
        if manual_reset:
            day = start_day + ep * day_step
            obs, state = env.manual_reset(day, hour, quarter)
        else:
            obs, state = env.reset()

        raw_obs = np.asarray(obs, dtype=np.float32)
        raw_state = np.asarray(state, dtype=np.float32)

        if obs_mean is not None and obs_std is not None:
            obs = _normalize_obs(raw_obs, obs_mean, obs_std)
        else:
            obs = raw_obs

        if state_mean is not None and state_std is not None:
            state = _normalize_state(raw_state, state_mean, state_std)
        else:
            state = raw_state

        done = False
        tracker = EpisodeMetricTracker(gamma=0.99, cost_limit=cost_limit)

        while not done:
            action = algo.select_action(
                obs,
                state=state,
                deterministic=False,
                num_candidates=20,
            ).reshape(-1)
            reward, done, info = env.step(action, add_noise=False)

            raw_next_obs = np.asarray(env.get_obs(), dtype=np.float32)
            raw_next_state = np.asarray(env.get_state(), dtype=np.float32)

            if obs_mean is not None and obs_std is not None:
                next_obs = _normalize_obs(raw_next_obs, obs_mean, obs_std)
            else:
                next_obs = raw_next_obs

            if state_mean is not None and state_std is not None:
                next_state = _normalize_state(raw_next_state, state_mean, state_std)
            else:
                next_state = raw_next_state

            tracker.update(reward=reward, info=info, env=env)

            obs, state = next_obs, next_state

        records.append(tracker.to_record())

    summary = summarize_records(records)
    return summary, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--state-dim", type=int, required=True)
    parser.add_argument("--act-dim", type=int, required=True)
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--action-low", type=float, default=-1.0)
    parser.add_argument("--action-high", type=float, default=1.0)
    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=730)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)
    parser.add_argument("--cost-limit", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--dataset-stats-path", type=str, default="")
    parser.add_argument("--actor-model", type=str, default="mlp", choices=["mlp", "residual"])
    parser.add_argument("--critic-model", type=str, default="mlp", choices=["mlp", "central", "maac"])
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--critic-attend-heads", type=int, default=4)
    args = parser.parse_args()

    summary, records = evaluate_mabcq_lag(
        model_path=args.model_path,
        obs_dim=args.obs_dim,
        state_dim=args.state_dim,
        act_dim=args.act_dim,
        n_agents=args.n_agents,
        action_low=args.action_low,
        action_high=args.action_high,
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
        cost_limit=args.cost_limit,
        device=args.device,
        dataset_stats_path=args.dataset_stats_path,
        actor_model_name=args.actor_model,
        critic_model_name=args.critic_model,
        actor_hidden_dims=tuple(args.actor_hidden_dims),
        critic_hidden_dims=tuple(args.critic_hidden_dims),
        critic_attend_heads=args.critic_attend_heads,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
