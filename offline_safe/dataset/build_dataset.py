# offline_safe/dataset/build_dataset.py
import os
import json
import argparse
import numpy as np
from tqdm import trange

from offline_safe.dataset.common import build_env, build_alg_args
from offline_safe.dataset.policies import RandomPolicy, NoisyDroopPolicy, ExpertMATD3Policy
from offline_safe.eval.safety_metrics import compute_step_cost


def build_cost(info):
    return compute_step_cost(info, destroy_w=10.0, v_out_w=1.0, q_loss_w=0.0)


def make_policy(args, env):
    if args.policy == "random":
        return RandomPolicy(env)

    if args.policy == "noisy_droop":
        return NoisyDroopPolicy(env, gain=args.droop_gain, noise_std=args.droop_noise_std)

    if args.policy == "expert_checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when policy=expert_checkpoint")
        alg_args = build_alg_args(
            alg=args.alg,
            env=env,
            scenario=args.scenario,
            episode_limit=args.episode_limit,
        )
        return ExpertMATD3Policy(
            args=alg_args,
            checkpoint_path=args.checkpoint,
            alg=args.alg,
        )

    raise ValueError(f"Unsupported policy: {args.policy}")


def reset_env(env, manual_reset=False, day=730, hour=23, quarter=2):
    if manual_reset:
        obs, state = env.manual_reset(day, hour, quarter)
    else:
        obs, state = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    return obs, state


def collect_one_episode(env, policy, add_noise=False, manual_reset=False, day=730, hour=23, quarter=2):
    obs, state = reset_env(
        env,
        manual_reset=manual_reset,
        day=day,
        hour=hour,
        quarter=quarter,
    )
    policy.reset()

    traj = {
        "obs": [],
        "state": [],
        "actions": [],
        "rewards": [],
        "costs": [],
        "next_obs": [],
        "next_state": [],
        "dones": [],
        "v_out": [],
        "q_loss": [],
        "destroy": [],
    }

    done = False
    while not done:
        actions = policy.act(obs, state, env).reshape(-1)  # env.step 接 joint action 向量

        reward, done, info = env.step(actions, add_noise=add_noise)

        next_obs = np.asarray(env.get_obs(), dtype=np.float32)
        next_state = np.asarray(env.get_state(), dtype=np.float32)

        cost = build_cost(info)
        v_out = float(info.get("percentage_of_v_out_of_control", 0.0))
        q_loss = float(info.get("q_loss", 0.0))
        destroy = float(info.get("destroy", 0.0))

        traj["obs"].append(obs)
        traj["state"].append(state)
        traj["actions"].append(actions.reshape(env.get_num_of_agents(), env.get_total_actions()))
        traj["rewards"].append([float(reward)])
        traj["costs"].append([float(cost)])
        traj["next_obs"].append(next_obs)
        traj["next_state"].append(next_state)
        traj["dones"].append([float(done)])
        traj["v_out"].append([v_out])
        traj["q_loss"].append([q_loss])
        traj["destroy"].append([destroy])

        obs, state = next_obs, next_state

    for k in traj:
        traj[k] = np.asarray(traj[k], dtype=np.float32)
    return traj


def save_episode_npz(save_path, traj):
    np.savez_compressed(save_path, **traj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="var_voltage_control")
    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--policy", type=str, required=True,
                        choices=["random", "noisy_droop", "expert_checkpoint"])
    parser.add_argument("--alg", type=str, default="matd3")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--add-noise", action="store_true")

    # for medium
    parser.add_argument("--droop-gain", type=float, default=6.0)
    parser.add_argument("--droop-noise-std", type=float, default=0.03)

    # optional deterministic reset
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=730)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env, env_cfg = build_env(
        env_name=args.env,
        scenario=args.scenario,
        mode=args.mode,
        voltage_barrier_type=args.voltage_barrier_type,
        episode_limit=args.episode_limit,
        seed=args.seed,
    )

    policy = make_policy(args, env)

    meta = {
        "env": args.env,
        "scenario": args.scenario,
        "mode": args.mode,
        "voltage_barrier_type": args.voltage_barrier_type,
        "episode_limit": args.episode_limit,
        "episodes": args.episodes,
        "policy": args.policy,
        "alg": args.alg,
        "checkpoint": args.checkpoint,
        "add_noise": args.add_noise,
        "manual_reset": args.manual_reset,
        "start_day": args.start_day,
        "day_step": args.day_step,
        "hour": args.hour,
        "quarter": args.quarter,
        "n_agents": env.get_num_of_agents(),
        "obs_size": env.get_obs_size(),
        "action_dim": env.get_total_actions(),
        "action_low": env.action_space.low,
        "action_high": env.action_space.high,
    }
    with open(os.path.join(args.save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for ep in trange(args.episodes, desc=f"Collect {args.policy}"):
        day = args.start_day + ep * args.day_step
        traj = collect_one_episode(
            env,
            policy,
            add_noise=args.add_noise,
            manual_reset=args.manual_reset,
            day=day,
            hour=args.hour,
            quarter=args.quarter,
        )
        save_episode_npz(os.path.join(args.save_dir, f"ep_{ep:05d}.npz"), traj)

    print(f"Done. Dataset saved to: {args.save_dir}")


if __name__ == "__main__":
    main()