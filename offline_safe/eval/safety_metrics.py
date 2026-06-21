from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

V_MIN = 0.95
V_MAX = 1.05
V_REF = 1.0


def to_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return float(default)
        return float(arr.mean()) if arr.ndim > 0 else float(arr.item())
    except Exception:
        return float(default)


def compute_step_cost(
    info: Dict,
    destroy_w: float = 10.0,
    v_out_w: float = 1.0,
    q_loss_w: float = 0.0,
) -> float:
    """Unified safety cost.

    Safety cost should focus on safety variables, not economic objectives.
    We therefore recommend using destroy + voltage-out-of-control as the default,
    and keep q_loss as an optional auxiliary term.
    """
    destroy = to_float(info.get("destroy", 0.0))
    v_out = to_float(info.get("percentage_of_v_out_of_control", 0.0))
    q_loss = to_float(info.get("q_loss", 0.0))
    return destroy_w * destroy + v_out_w * v_out + q_loss_w * q_loss


def extract_step_signals(info: Dict) -> Dict[str, float]:
    return {
        "destroy": to_float(info.get("destroy", 0.0)),
        "v_out": to_float(info.get("percentage_of_v_out_of_control", 0.0)),
        "q_loss": to_float(info.get("q_loss", 0.0)),
    }


def compute_voltage_metrics_from_bus(
    bus_voltage,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    v_ref: float = V_REF,
) -> Dict[str, float]:
    v = np.asarray(bus_voltage, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return {
            "controllable": 1.0,
            "avg_voltage_dev": 0.0,
            "max_v_drop_dev": 0.0,
            "max_v_rise_dev": 0.0,
        }

    drop = np.maximum(0.0, v_min - v)
    rise = np.maximum(0.0, v - v_max)
    return {
        "controllable": float(np.all((v >= v_min) & (v <= v_max))),
        "avg_voltage_dev": float(np.mean(np.abs(v - v_ref))),
        "max_v_drop_dev": float(np.max(drop)),
        "max_v_rise_dev": float(np.max(rise)),
    }


def extract_power_loss(env=None, info: Optional[Dict] = None) -> float:
    if info is not None:
        for key in ["power_loss", "line_loss", "total_line_loss", "pl"]:
            if key in info:
                return to_float(info[key])
    if env is not None and hasattr(env, "_get_res_line_loss"):
        try:
            raw = env._get_res_line_loss()
            arr = np.asarray(raw, dtype=np.float32)
            return float(arr.sum())
        except Exception:
            return 0.0
    return 0.0


@dataclass
class EpisodeMetricTracker:
    gamma: float = 0.99
    cost_limit: Optional[float] = None
    destroy_w: float = 10.0
    v_out_w: float = 1.0
    q_loss_w: float = 0.0

    episode_return: float = 0.0
    episode_cost: float = 0.0
    discounted_episode_cost: float = 0.0
    length: int = 0
    v_outs: List[float] = field(default_factory=list)
    destroys: List[float] = field(default_factory=list)
    q_losses: List[float] = field(default_factory=list)
    controllable_steps: List[float] = field(default_factory=list)
    voltage_devs: List[float] = field(default_factory=list)
    max_v_drop_devs: List[float] = field(default_factory=list)
    max_v_rise_devs: List[float] = field(default_factory=list)
    power_losses: List[float] = field(default_factory=list)

    def update(self, reward: float, info: Dict, env=None):
        signals = extract_step_signals(info)
        step_cost = compute_step_cost(
            info,
            destroy_w=self.destroy_w,
            v_out_w=self.v_out_w,
            q_loss_w=self.q_loss_w,
        )
        self.episode_return += float(reward)
        self.episode_cost += float(step_cost)
        self.discounted_episode_cost += (self.gamma ** self.length) * float(step_cost)
        self.length += 1

        self.v_outs.append(signals["v_out"])
        self.destroys.append(signals["destroy"])
        self.q_losses.append(signals["q_loss"])
        self.power_losses.append(extract_power_loss(env=env, info=info))

        if env is not None and hasattr(env, "_get_res_bus_v"):
            try:
                step_voltage_metrics = compute_voltage_metrics_from_bus(env._get_res_bus_v())
            except Exception:
                step_voltage_metrics = None
        else:
            step_voltage_metrics = None

        if step_voltage_metrics is not None:
            self.controllable_steps.append(step_voltage_metrics["controllable"])
            self.voltage_devs.append(step_voltage_metrics["avg_voltage_dev"])
            self.max_v_drop_devs.append(step_voltage_metrics["max_v_drop_dev"])
            self.max_v_rise_devs.append(step_voltage_metrics["max_v_rise_dev"])
        else:
            controllable = 1.0 if signals["v_out"] <= 1e-12 else 0.0
            self.controllable_steps.append(controllable)
            self.voltage_devs.append(0.0)
            self.max_v_drop_devs.append(0.0)
            self.max_v_rise_devs.append(0.0)

    def to_record(self) -> Dict[str, float]:
        constraint_satisfied = None
        cost_violation_ratio = None
        if self.cost_limit is not None and self.cost_limit > 0:
            constraint_satisfied = float(self.discounted_episode_cost <= self.cost_limit)
            cost_violation_ratio = float(max(0.0, self.discounted_episode_cost - self.cost_limit) / self.cost_limit)

        return {
            "episode_return": float(self.episode_return),
            "episode_cost": float(self.episode_cost),
            "discounted_episode_cost": float(self.discounted_episode_cost),
            "avg_v_out": float(np.mean(self.v_outs)) if self.v_outs else 0.0,
            "sum_destroy": float(np.sum(self.destroys)) if self.destroys else 0.0,
            "avg_q_loss": float(np.mean(self.q_losses)) if self.q_losses else 0.0,
            "length": int(self.length),
            "cr": float(np.mean(self.controllable_steps)) if self.controllable_steps else 0.0,
            "pl": float(np.mean(self.power_losses)) if self.power_losses else 0.0,
            "v_dev": float(np.mean(self.voltage_devs)) if self.voltage_devs else 0.0,
            "max_v_drop_dev": float(np.mean(self.max_v_drop_devs)) if self.max_v_drop_devs else 0.0,
            "max_v_rise_dev": float(np.mean(self.max_v_rise_devs)) if self.max_v_rise_devs else 0.0,
            "constraint_satisfied": constraint_satisfied,
            "cost_violation_ratio": cost_violation_ratio,
            "no_destroy": float(np.sum(self.destroys) <= 1e-12),
        }


def summarize_records(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {"eval_episodes": 0}

    def _mean(key: str, default: float = 0.0):
        vals = [x[key] for x in records if key in x and x[key] is not None]
        return float(np.mean(vals)) if vals else float(default)

    def _mean_or_none(key: str):
        vals = [x[key] for x in records if key in x and x[key] is not None]
        return float(np.mean(vals)) if vals else None

    def _cvar(key: str, alpha: float = 0.95):
        vals = np.asarray([x[key] for x in records if key in x and x[key] is not None], dtype=np.float32)
        if vals.size == 0:
            return 0.0
        var = np.quantile(vals, alpha)
        tail = vals[vals >= var]
        return float(tail.mean()) if tail.size > 0 else float(var)

    return {
        "eval_episodes": len(records),
        "avg_return": _mean("episode_return"),
        "avg_cost": _mean("episode_cost"),
        "avg_discounted_cost": _mean("discounted_episode_cost"),
        "avg_v_out": _mean("avg_v_out"),
        "avg_destroy": _mean("sum_destroy"),
        "avg_q_loss": _mean("avg_q_loss"),
        "avg_length": _mean("length"),
        "cr": _mean("cr"),
        "pl": _mean("pl"),
        "v_dev": _mean("v_dev"),
        "max_v_drop_dev": _mean("max_v_drop_dev"),
        "max_v_rise_dev": _mean("max_v_rise_dev"),
        "constraint_satisfaction_rate": _mean_or_none("constraint_satisfied"),
        "mean_cost_violation_ratio": _mean("cost_violation_ratio"),
        "no_destroy_rate": _mean("no_destroy"),
        "worst_episode_cost": float(max(x.get("episode_cost", 0.0) for x in records)),
        "worst_episode_v_out": float(max(x.get("avg_v_out", 0.0) for x in records)),
        "cvar95_cost": _cvar("episode_cost", alpha=0.95),
    }