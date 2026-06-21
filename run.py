# -*- coding: utf-8 -*-
"""
单文件运行入口（多模型批量版）：
在 PyCharm 中只需修改 CONFIG 区域，然后运行本文件即可。

修改点：
1. 支持多窗口采样 start_days
2. poor / medium / good 都可跨多个时间窗口采集
3. replay meta 中记录 start_days
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import glob
import shutil
import random
import yaml
import numpy as np
import torch
from tqdm import trange

from offline_safe.eval.safety_metrics import compute_step_cost
from utilities.util import convert, prep_obs, translate_action
from models.model_registry import Model
from environments.var_voltage_control.voltage_control_env import VoltageControl
from baseline_checkpoints import SUPPORTED_BASELINE_MODELS, build_default_checkpoint_map


CONFIG = {
    "run_mode": "mix_replay_full",

    # 支持：matd3, maddpg, sqddpg, iddpg, maac, iac, ippo, mappo, coma, facmaddpg
    "models": ["matd3", "maddpg"],

    # checkpoint 唯一来源参数
    "baseline_save_root": "./",
    "baseline_alias": "0",

    "env_name": "var_voltage_control",
    "scenario": "case33_3min_final",
    "mode": "distributed",
    "voltage_barrier_type": "l1",
    "episode_limit": 480,
    "seed": 1,

    "dataset_root": "./offline_safe/data/offline_case33_all",

    "full_episodes": 200,

    "manual_reset": True,

    # 单窗口兼容参数（如果 start_days=None 时使用）
    "start_day": 730,
    "day_step": 1,

    # 多窗口采样：推荐直接用这个
    "start_days": [730, 760, 790, 820, 850],

    "hour": 23,
    "quarter": 2,

    "droop_gain": 4.0,
    "droop_noise_std": 0.02,

    "add_noise": False,

    "replay_total_full": 300,
    "poor_ratio": 0.1,
    "medium_ratio": 0.5,
    "good_ratio": 0.4,
    "replay_seed": 1,
}


ACTION_SCALE_MAP = {
    "case33_3min_final": 0.8,
    "case141_3min_final": 0.6,
    "case322_3min_final": 0.8,
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def list_eps(root: str):
    return sorted(glob.glob(os.path.join(root, "ep_*.npz")))


def summarize_dataset_dir(dataset_dir, split_name="replay"):
    files = list_eps(dataset_dir)
    if len(files) == 0:
        raise ValueError(f"{split_name} 目录为空：{dataset_dir}")

    episode_returns = []
    episode_costs = []
    avg_v_outs = []
    sum_destroys = []
    lengths = []

    for fp in files:
        d = np.load(fp)
        episode_returns.append(float(d["rewards"].sum()))
        episode_costs.append(float(d["costs"].sum()))
        avg_v_outs.append(float(d["v_out"].mean()) if "v_out" in d else 0.0)
        sum_destroys.append(float(d["destroy"].sum()) if "destroy" in d else 0.0)
        lengths.append(int(len(d["rewards"])))

    return {
        "split": split_name,
        "num_episodes": len(files),
        "avg_return": float(np.mean(episode_returns)),
        "avg_cost": float(np.mean(episode_costs)),
        "avg_v_out": float(np.mean(avg_v_outs)),
        "avg_destroy": float(np.mean(sum_destroys)),
        "avg_length": float(np.mean(lengths)),
    }


def iter_model_cfgs(base_cfg):
    checkpoint_map = build_default_checkpoint_map(
        save_root=base_cfg["baseline_save_root"],
        scenario=base_cfg["scenario"],
        mode=base_cfg["mode"],
        voltage_barrier_type=base_cfg["voltage_barrier_type"],
        alias=base_cfg["baseline_alias"],
    )

    for alg in base_cfg["models"]:
        if alg not in SUPPORTED_BASELINE_MODELS:
            print(f"[skip] {alg}: unsupported baseline model")
            continue

        checkpoint_path = checkpoint_map.get(alg, "")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"[skip] {alg}: checkpoint 不存在 -> {checkpoint_path}")
            continue

        cfg = dict(base_cfg)
        cfg["alg"] = alg
        cfg["checkpoint_path"] = checkpoint_path
        cfg["save_root"] = os.path.join(base_cfg["dataset_root"], alg)
        yield alg, cfg


def build_env_config(
    env_name="var_voltage_control",
    scenario="case33_3min_final",
    mode="distributed",
    voltage_barrier_type="l1",
    episode_limit=480,
    seed=1,
):
    with open(f"./args/env_args/{env_name}.yaml", "r", encoding="utf-8") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]

    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = scenario
    env_config_dict["data_path"] = "/".join(data_path)

    env_config_dict["mode"] = mode
    env_config_dict["voltage_barrier_type"] = voltage_barrier_type
    env_config_dict["episode_limit"] = episode_limit
    env_config_dict["seed"] = seed
    env_config_dict["action_scale"] = ACTION_SCALE_MAP[scenario]
    env_config_dict["action_bias"] = 0.0
    return env_config_dict


def build_env(cfg: dict):
    env_cfg = build_env_config(
        env_name=cfg["env_name"],
        scenario=cfg["scenario"],
        mode=cfg["mode"],
        voltage_barrier_type=cfg["voltage_barrier_type"],
        episode_limit=cfg["episode_limit"],
        seed=cfg["seed"],
    )
    env = VoltageControl(env_cfg)
    return env, env_cfg


def build_alg_args(cfg: dict, env):
    with open("./args/default.yaml", "r", encoding="utf-8") as f:
        default_config_dict = yaml.safe_load(f)

    with open(f"./args/alg_args/{cfg['alg']}.yaml", "r", encoding="utf-8") as f:
        alg_config_dict = yaml.safe_load(f)["alg_args"]

    default_config_dict["max_steps"] = cfg["episode_limit"]
    alg_config_dict["action_scale"] = ACTION_SCALE_MAP[cfg["scenario"]]
    alg_config_dict["action_bias"] = 0.0
    alg_config_dict["agent_num"] = env.get_num_of_agents()
    alg_config_dict["obs_size"] = env.get_obs_size()
    alg_config_dict["action_dim"] = env.get_total_actions()
    alg_config_dict["cuda"] = torch.cuda.is_available()

    merged = {**default_config_dict, **alg_config_dict}
    return convert(merged)


class RandomPolicy:
    def __init__(self, env):
        self.low = np.asarray(env.action_space.low, dtype=np.float32)
        self.high = np.asarray(env.action_space.high, dtype=np.float32)
        self.n_agents = env.get_num_of_agents()
        self.action_dim = env.get_total_actions()

    def reset(self):
        pass

    def act(self, obs, state, env):
        a = np.random.uniform(
            low=self.low,
            high=self.high,
            size=(self.n_agents, self.action_dim),
        ).astype(np.float32)
        return a


class NoisyDroopPolicy:
    def __init__(self, env, gain=6.0, noise_std=0.03):
        self.low = np.asarray(env.action_space.low, dtype=np.float32)
        self.high = np.asarray(env.action_space.high, dtype=np.float32)
        self.n_agents = env.get_num_of_agents()
        self.action_dim = env.get_total_actions()
        self.gain = gain
        self.noise_std = noise_std

    def reset(self):
        pass

    def act(self, obs, state, env):
        bus_v = env._get_voltage()
        sgen_bus = env.powergrid.sgen["bus"].to_numpy(copy=True)
        local_v = bus_v[sgen_bus]

        action = self.gain * (1.0 - local_v)
        action += np.random.normal(0.0, self.noise_std, size=action.shape)
        action = np.clip(action, self.low, self.high)
        return action.reshape(self.n_agents, self.action_dim).astype(np.float32)


class ExpertCheckpointPolicy:
    def __init__(self, args, checkpoint_path, alg="matd3"):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_cls = Model[alg]
        if args.target:
            target_net = model_cls(args)
            self.behaviour_net = model_cls(args, target_net)
        else:
            self.behaviour_net = model_cls(args)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.behaviour_net.load_state_dict(checkpoint["model_state_dict"])
        self.behaviour_net.to(self.device)
        self.behaviour_net.eval()

        self.last_hid = None

    def reset(self):
        self.last_hid = self.behaviour_net.policy_dicts[0].init_hidden()

    @torch.no_grad()
    def act(self, obs, state, env):
        state_tensor = prep_obs(obs).contiguous().view(
            1, self.args.agent_num, self.args.obs_size
        ).to(self.device)

        action, _, _, _, hid = self.behaviour_net.get_actions(
            state_tensor,
            status="test",
            exploration=False,
            actions_avail=torch.tensor(env.get_avail_actions(), device=self.device),
            target=False,
            last_hid=self.last_hid,
        )

        _, actual = translate_action(self.args, action, env)
        self.last_hid = hid

        actual = np.asarray(actual, dtype=np.float32).reshape(
            self.args.agent_num, self.args.action_dim
        )
        return actual


def build_cost(info: dict) -> float:
    return compute_step_cost(info, destroy_w=10.0, v_out_w=1.0, q_loss_w=0.0)


def get_episode_day(cfg, ep_idx: int) -> int:
    start_days = cfg.get("start_days", None)
    if start_days is None or len(start_days) == 0:
        return cfg["start_day"] + ep_idx * cfg["day_step"]

    window_idx = ep_idx % len(start_days)
    cycle_idx = ep_idx // len(start_days)
    return int(start_days[window_idx] + cycle_idx * cfg["day_step"])


def reset_env(env, cfg, day=None):
    if cfg["manual_reset"]:
        if day is None:
            day = cfg["start_day"]
        obs, state = env.manual_reset(day, cfg["hour"], cfg["quarter"])
    else:
        obs, state = env.reset()

    obs = np.asarray(obs, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    return obs, state


def collect_one_episode(env, policy, cfg, day=None):
    obs, state = reset_env(env, cfg, day=day)
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
    ep_return = 0.0
    ep_cost = 0.0

    while not done:
        actions = policy.act(obs, state, env)
        joint_action = actions.reshape(-1)

        reward, done, info = env.step(joint_action, add_noise=cfg["add_noise"])

        next_obs = np.asarray(env.get_obs(), dtype=np.float32)
        next_state = np.asarray(env.get_state(), dtype=np.float32)

        cost = build_cost(info)
        v_out = float(info.get("percentage_of_v_out_of_control", 0.0))
        q_loss = float(info.get("q_loss", 0.0))
        destroy = float(info.get("destroy", 0.0))

        traj["obs"].append(obs)
        traj["state"].append(state)
        traj["actions"].append(actions)
        traj["rewards"].append([float(reward)])
        traj["costs"].append([float(cost)])
        traj["next_obs"].append(next_obs)
        traj["next_state"].append(next_state)
        traj["dones"].append([float(done)])
        traj["v_out"].append([v_out])
        traj["q_loss"].append([q_loss])
        traj["destroy"].append([destroy])

        ep_return += float(reward)
        ep_cost += float(cost)

        obs, state = next_obs, next_state

    for k in traj:
        traj[k] = np.asarray(traj[k], dtype=np.float32)

    summary = {
        "episode_return": ep_return,
        "episode_cost": ep_cost,
        "avg_v_out": float(traj["v_out"].mean()) if len(traj["v_out"]) > 0 else 0.0,
        "sum_destroy": float(traj["destroy"].sum()) if len(traj["destroy"]) > 0 else 0.0,
        "length": int(len(traj["rewards"])),
    }
    return traj, summary


def save_episode_npz(save_path, traj):
    np.savez_compressed(save_path, **traj)


def make_policy_for_split(split_name, env, cfg):
    if split_name == "poor":
        return RandomPolicy(env)

    if split_name == "medium":
        return NoisyDroopPolicy(
            env,
            gain=cfg["droop_gain"],
            noise_std=cfg["droop_noise_std"],
        )

    if split_name == "good":
        alg_args = build_alg_args(cfg, env)
        return ExpertCheckpointPolicy(
            args=alg_args,
            checkpoint_path=cfg["checkpoint_path"],
            alg=cfg["alg"],
        )

    raise ValueError(f"Unknown split: {split_name}")


def collect_split(split_name, episodes, cfg):
    env, _ = build_env(cfg)
    policy = make_policy_for_split(split_name, env, cfg)

    split_dir = os.path.join(cfg["save_root"], split_name)
    ensure_dir(split_dir)

    meta = {
        "split": split_name,
        "env_name": cfg["env_name"],
        "scenario": cfg["scenario"],
        "mode": cfg["mode"],
        "voltage_barrier_type": cfg["voltage_barrier_type"],
        "episode_limit": cfg["episode_limit"],
        "episodes": episodes,
        "manual_reset": cfg["manual_reset"],
        "start_day": cfg["start_day"],
        "start_days": cfg.get("start_days", None),
        "day_step": cfg["day_step"],
        "hour": cfg["hour"],
        "quarter": cfg["quarter"],
        "add_noise": cfg["add_noise"],
        "seed": cfg["seed"],
        "alg": cfg["alg"],
        "checkpoint_path": cfg["checkpoint_path"] if split_name == "good" else "",
        "droop_gain": cfg["droop_gain"] if split_name == "medium" else None,
        "droop_noise_std": cfg["droop_noise_std"] if split_name == "medium" else None,
        "n_agents": env.get_num_of_agents(),
        "obs_size": env.get_obs_size(),
        "action_dim": env.get_total_actions(),
        "action_low": np.asarray(env.action_space.low).tolist(),
        "action_high": np.asarray(env.action_space.high).tolist(),
    }
    save_json(os.path.join(split_dir, "meta.json"), meta)

    summaries = []
    print(f"\n[Collect {cfg['alg']}::{split_name}] save_dir = {split_dir}")
    print(f"episodes = {episodes}\n")

    for ep in trange(episodes, desc=f"Collect {cfg['alg']}::{split_name}"):
        day = get_episode_day(cfg, ep)
        traj, summary = collect_one_episode(env, policy, cfg, day=day)
        save_episode_npz(os.path.join(split_dir, f"ep_{ep:05d}.npz"), traj)
        summaries.append(summary)

    summary_json = {
        "split": split_name,
        "num_episodes": len(summaries),
        "avg_return": float(np.mean([s["episode_return"] for s in summaries])),
        "avg_cost": float(np.mean([s["episode_cost"] for s in summaries])),
        "avg_v_out": float(np.mean([s["avg_v_out"] for s in summaries])),
        "avg_destroy": float(np.mean([s["sum_destroy"] for s in summaries])),
        "avg_length": float(np.mean([s["length"] for s in summaries])),
    }
    save_json(os.path.join(split_dir, "summary.json"), summary_json)
    return summary_json


def mix_replay(total_episodes, cfg):
    poor_dir = os.path.join(cfg["save_root"], "poor")
    medium_dir = os.path.join(cfg["save_root"], "medium")
    good_dir = os.path.join(cfg["save_root"], "good")
    replay_dir = os.path.join(cfg["save_root"], "replay")
    ensure_dir(replay_dir)

    poor_files = list_eps(poor_dir)
    medium_files = list_eps(medium_dir)
    good_files = list_eps(good_dir)

    if len(poor_files) == 0:
        raise ValueError(f"poor 目录为空：{poor_dir}")
    if len(medium_files) == 0:
        raise ValueError(f"medium 目录为空：{medium_dir}")
    if len(good_files) == 0:
        raise ValueError(f"good 目录为空：{good_dir}")

    rng = random.Random(cfg["replay_seed"])

    n_poor = int(total_episodes * cfg["poor_ratio"])
    n_medium = int(total_episodes * cfg["medium_ratio"])
    n_good = total_episodes - n_poor - n_medium

    if len(poor_files) < n_poor:
        raise ValueError(f"poor 数据不够：需要 {n_poor} 个，当前只有 {len(poor_files)} 个")
    if len(medium_files) < n_medium:
        raise ValueError(f"medium 数据不够：需要 {n_medium} 个，当前只有 {len(medium_files)} 个")
    if len(good_files) < n_good:
        raise ValueError(f"good 数据不够：需要 {n_good} 个，当前只有 {len(good_files)} 个")

    for fp in list_eps(replay_dir):
        os.remove(fp)

    rng.shuffle(poor_files)
    rng.shuffle(medium_files)
    rng.shuffle(good_files)

    selected = []
    selected += [("poor", fp) for fp in poor_files[:n_poor]]
    selected += [("medium", fp) for fp in medium_files[:n_medium]]
    selected += [("good", fp) for fp in good_files[:n_good]]
    rng.shuffle(selected)

    manifest = []
    for idx, (src_type, src_fp) in enumerate(selected):
        dst_fp = os.path.join(replay_dir, f"ep_{idx:05d}.npz")
        shutil.copy2(src_fp, dst_fp)
        manifest.append({
            "dst": os.path.basename(dst_fp),
            "src_type": src_type,
            "src_path": src_fp,
        })

    save_json(os.path.join(replay_dir, "manifest.json"), {
        "total_episodes": total_episodes,
        "poor_ratio": cfg["poor_ratio"],
        "medium_ratio": cfg["medium_ratio"],
        "good_ratio": cfg["good_ratio"],
        "manifest": manifest,
    })

    env, _ = build_env(cfg)
    replay_meta = {
        "split": "replay",
        "source_type": "mixed_replay",
        "alg": cfg["alg"],
        "checkpoint_path": cfg["checkpoint_path"],
        "env_name": cfg["env_name"],
        "scenario": cfg["scenario"],
        "mode": cfg["mode"],
        "voltage_barrier_type": cfg["voltage_barrier_type"],
        "episode_limit": cfg["episode_limit"],
        "total_episodes": total_episodes,
        "manual_reset": cfg["manual_reset"],
        "start_day": cfg["start_day"],
        "start_days": cfg.get("start_days", None),
        "day_step": cfg["day_step"],
        "hour": cfg["hour"],
        "quarter": cfg["quarter"],
        "seed": cfg["seed"],
        "n_agents": env.get_num_of_agents(),
        "obs_size": env.get_obs_size(),
        "action_dim": env.get_total_actions(),
        "action_low": np.asarray(env.action_space.low).tolist(),
        "action_high": np.asarray(env.action_space.high).tolist(),
    }
    save_json(os.path.join(replay_dir, "meta.json"), replay_meta)

    replay_summary = summarize_dataset_dir(replay_dir, split_name="replay")
    save_json(os.path.join(replay_dir, "summary.json"), replay_summary)
    return replay_summary


def test_expert(cfg):
    env, _ = build_env(cfg)
    alg_args = build_alg_args(cfg, env)
    policy = ExpertCheckpointPolicy(
        args=alg_args,
        checkpoint_path=cfg["checkpoint_path"],
        alg=cfg["alg"],
    )

    day = get_episode_day(cfg, 0)
    _, summary = collect_one_episode(env, policy, cfg, day=day)

    print(f"\n[Expert Test Result::{cfg['alg']}]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def collect_and_mix_full_for_model(cfg):
    ensure_dir(cfg["save_root"])

    poor_summary = collect_split("poor", cfg["full_episodes"], cfg)
    medium_summary = collect_split("medium", cfg["full_episodes"], cfg)
    good_summary = collect_split("good", cfg["full_episodes"], cfg)
    replay_summary = mix_replay(cfg["replay_total_full"], cfg)

    all_summary = {
        "alg": cfg["alg"],
        "poor": poor_summary,
        "medium": medium_summary,
        "good": good_summary,
        "replay": replay_summary,
    }
    save_json(os.path.join(cfg["save_root"], "all_summary.json"), all_summary)
    return all_summary


def main():
    run_mode = CONFIG["run_mode"]
    ensure_dir(CONFIG["dataset_root"])

    print("=" * 70)
    print("RUN MODE:", run_mode)
    print("SCENARIO:", CONFIG["scenario"])
    print("DATASET ROOT:", CONFIG["dataset_root"])
    print("MODELS:", CONFIG["models"])
    print("START DAYS:", CONFIG.get("start_days", None))
    print("=" * 70)

    if run_mode == "test_expert":
        for alg, cfg in iter_model_cfgs(CONFIG):
            test_expert(cfg)
        return

    if run_mode in ["collect_all_full", "mix_replay_full"]:
        global_summary = {}

        for alg, cfg in iter_model_cfgs(CONFIG):
            print(f"\n{'=' * 20} FULL PIPELINE FOR {alg} {'=' * 20}")
            model_summary = collect_and_mix_full_for_model(cfg)
            global_summary[alg] = model_summary

            print(f"\n[{alg}] poor summary:")
            print(json.dumps(model_summary["poor"], indent=2, ensure_ascii=False))
            print(f"\n[{alg}] medium summary:")
            print(json.dumps(model_summary["medium"], indent=2, ensure_ascii=False))
            print(f"\n[{alg}] good summary:")
            print(json.dumps(model_summary["good"], indent=2, ensure_ascii=False))
            print(f"\n[{alg}] replay summary:")
            print(json.dumps(model_summary["replay"], indent=2, ensure_ascii=False))

        save_json(os.path.join(CONFIG["dataset_root"], "global_summary.json"), global_summary)
        print(f"\nDone. Global summary saved to: {os.path.join(CONFIG['dataset_root'], 'global_summary.json')}")
        return

    raise ValueError(f"Unsupported run_mode: {run_mode}")


if __name__ == "__main__":
    main()