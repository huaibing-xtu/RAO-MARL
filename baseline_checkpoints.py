import os

SUPPORTED_BASELINE_MODELS = [
    "maddpg", "sqddpg", "iac", "iddpg", "coma",
    "maac", "matd3", "ippo", "mappo", "facmaddpg"
]

DEFAULT_SAVE_ROOT = "./"
DEFAULT_ENV = "var_voltage_control"
DEFAULT_SCENARIO = "case33_3min_final"
DEFAULT_MODE = "distributed"
DEFAULT_VOLTAGE_BARRIER_TYPE = "l1"
DEFAULT_ALIAS = "0"


def build_baseline_log_name(
    env: str = DEFAULT_ENV,
    scenario: str = DEFAULT_SCENARIO,
    mode: str = DEFAULT_MODE,
    alg: str = "matd3",
    voltage_barrier_type: str = DEFAULT_VOLTAGE_BARRIER_TYPE,
    alias: str = DEFAULT_ALIAS,
) -> str:
    return "-".join([env, scenario, mode, alg, voltage_barrier_type, alias])


def build_baseline_checkpoint_path(
    save_root: str = DEFAULT_SAVE_ROOT,
    env: str = DEFAULT_ENV,
    scenario: str = DEFAULT_SCENARIO,
    mode: str = DEFAULT_MODE,
    alg: str = "matd3",
    voltage_barrier_type: str = DEFAULT_VOLTAGE_BARRIER_TYPE,
    alias: str = DEFAULT_ALIAS,
) -> str:
    log_name = build_baseline_log_name(
        env=env,
        scenario=scenario,
        mode=mode,
        alg=alg,
        voltage_barrier_type=voltage_barrier_type,
        alias=alias,
    )
    return os.path.join(save_root, "model_save", log_name, "model.pt")


def build_default_checkpoint_map(
    save_root: str = DEFAULT_SAVE_ROOT,
    env: str = DEFAULT_ENV,
    scenario: str = DEFAULT_SCENARIO,
    mode: str = DEFAULT_MODE,
    voltage_barrier_type: str = DEFAULT_VOLTAGE_BARRIER_TYPE,
    alias: str = DEFAULT_ALIAS,
):
    return {
        alg: build_baseline_checkpoint_path(
            save_root=save_root,
            env=env,
            scenario=scenario,
            mode=mode,
            alg=alg,
            voltage_barrier_type=voltage_barrier_type,
            alias=alias,
        )
        for alg in SUPPORTED_BASELINE_MODELS
    }