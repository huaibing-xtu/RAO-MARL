# offline_safe/dataset/common.py
import os
import yaml

from utilities.util import convert
from environments.var_voltage_control.voltage_control_env import VoltageControl

ACTION_SCALE_MAP = {
    "case33_3min_final": 0.8,
    "case141_3min_final": 0.6,
    "case322_3min_final": 0.8,
}


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

    if scenario not in ACTION_SCALE_MAP:
        raise ValueError(f"Unsupported scenario: {scenario}")

    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = ACTION_SCALE_MAP[scenario]
    return env_config_dict


def build_env(
    env_name="var_voltage_control",
    scenario="case33_3min_final",
    mode="distributed",
    voltage_barrier_type="l1",
    episode_limit=480,
    seed=1,
):
    env_cfg = build_env_config(
        env_name=env_name,
        scenario=scenario,
        mode=mode,
        voltage_barrier_type=voltage_barrier_type,
        episode_limit=episode_limit,
        seed=seed,
    )
    env = VoltageControl(env_cfg)
    return env, env_cfg


def build_alg_args(
    alg="matd3",
    env=None,
    scenario="case33_3min_final",
    episode_limit=480,
):
    with open("./args/default.yaml", "r", encoding="utf-8") as f:
        default_config_dict = yaml.safe_load(f)

    with open(f"./args/alg_args/{alg}.yaml", "r", encoding="utf-8") as f:
        alg_config_dict = yaml.safe_load(f)["alg_args"]

    default_config_dict["max_steps"] = episode_limit
    alg_config_dict["action_scale"] = ACTION_SCALE_MAP[scenario]
    alg_config_dict["action_bias"] = 0.0

    if env is None:
        raise ValueError("env must be provided to infer agent_num / obs_size / action_dim")

    alg_config_dict["agent_num"] = env.get_num_of_agents()
    alg_config_dict["obs_size"] = env.get_obs_size()
    alg_config_dict["action_dim"] = env.get_total_actions()
    alg_config_dict["cuda"] = False

    merged = {**default_config_dict, **alg_config_dict}
    return convert(merged)