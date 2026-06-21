
import argparse
import json
import os
from typing import Dict, List
from typing import Optional
import numpy as np
import torch
import yaml

from environments.var_voltage_control.voltage_control_env import VoltageControl
from models.model_registry import Model, Strategy
from offline_safe.eval.safety_metrics import EpisodeMetricTracker, summarize_records
from utilities.util import convert, prep_obs, translate_action


def build_env_and_args(alg: str, scenario: str, mode: str, voltage_barrier_type: str, episode_limit: int):
    with open("./args/env_args/var_voltage_control.yaml", "r", encoding="utf-8") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = scenario
    env_config_dict["data_path"] = "/".join(data_path)
    env_config_dict["mode"] = mode
    env_config_dict["voltage_barrier_type"] = voltage_barrier_type
    env_config_dict["episode_limit"] = episode_limit
    if scenario == "case33_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    elif scenario == "case141_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.6
    elif scenario == "case322_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    env = VoltageControl(env_config_dict)

    with open("./args/default.yaml", "r", encoding="utf-8") as f:
        default_config_dict = yaml.safe_load(f)
    default_config_dict["max_steps"] = episode_limit

    with open(f"./args/alg_args/{alg}.yaml", "r", encoding="utf-8") as f:
        alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]
    alg_config_dict["agent_num"] = env.get_num_of_agents()
    alg_config_dict["obs_size"] = env.get_obs_size()
    alg_config_dict["action_dim"] = env.get_total_actions()
    alg_config_dict["cuda"] = False
    args = convert({**default_config_dict, **alg_config_dict})
    return env, args


def build_model(alg: str, args, model_path: str):
    model_cls = Model[alg]
    if getattr(args, "target", False):
        target_net = model_cls(args)
        behaviour_net = model_cls(args, target_net)
    else:
        behaviour_net = model_cls(args)
    checkpoint = torch.load(model_path, map_location="cpu")
    behaviour_net.load_state_dict(checkpoint["model_state_dict"])
    behaviour_net.eval()
    return behaviour_net

def resolve_cost_limit(cost_limit:  Optional[float], default_cost_limit: float = 5.6) -> float:
    if cost_limit is None:
        return float(default_cost_limit)
    return float(cost_limit)

def evaluate_baseline(
    model_path: str,
    alg: str,
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
    cost_limit: Optional[float] = None,
) -> tuple[Dict, List[Dict]]:
    env, args = build_env_and_args(alg, scenario, mode, voltage_barrier_type, episode_limit)
    resolved_cost_limit = resolve_cost_limit(cost_limit)

    if Strategy[alg] != "pg":
        raise NotImplementedError(f"Only pg strategy is currently supported, got {Strategy[alg]}")

    model = build_model(alg, args, model_path)
    device = torch.device("cpu")
    records: List[Dict] = []

    for ep in range(eval_episodes):
        if manual_reset:
            state, _ = env.manual_reset(start_day + ep * day_step, hour, quarter)
        else:
            state, _ = env.reset()

        last_hid = model.policy_dicts[0].init_hidden()
        tracker = EpisodeMetricTracker(gamma=0.99, cost_limit=resolved_cost_limit)

        done = False
        t = 0
        while not done and t < episode_limit:
            state_t = prep_obs(state).contiguous().view(1, args.agent_num, args.obs_size).to(device)
            action, _, _, _, hid = model.get_actions(
                state_t,
                status="test",
                exploration=False,
                actions_avail=torch.tensor(env.get_avail_actions()),
                target=False,
                last_hid=last_hid,
            )
            _, actual = translate_action(args, action, env)
            reward, done, info = env.step(actual, add_noise=False)
            tracker.update(reward=reward, info=info, env=env)
            state = env.get_obs()
            last_hid = hid
            t += 1

        records.append(tracker.to_record())

    summary = summarize_records(records)
    summary["cost_limit"] = float(resolved_cost_limit)
    return summary, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--alg", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    parser.add_argument("--episode-limit", type=int, default=480)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--manual-reset", action="store_true")
    parser.add_argument("--start-day", type=int, default=730)
    parser.add_argument("--day-step", type=int, default=1)
    parser.add_argument("--hour", type=int, default=23)
    parser.add_argument("--quarter", type=int, default=2)
    parser.add_argument("--cost-limit", type=float, default=None)
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    summary, records = evaluate_baseline(
        model_path=args.model_path,
        alg=args.alg,
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
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
