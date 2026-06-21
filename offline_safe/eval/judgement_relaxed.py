from typing import Dict, List, Optional

DEFAULT_KEY_METRICS = {
    "performance": {
        "avg_return": {"direction": "higher", "weight": 0.55},
        "cr": {"direction": "higher", "weight": 0.15},
        "pl": {"direction": "lower", "weight": 0.15},
        "avg_q_loss": {"direction": "lower", "weight": 0.15},
    },
    "safety": {
        "avg_cost": {"direction": "lower", "weight": 0.30},
        "avg_v_out": {"direction": "lower", "weight": 0.20},
        "worst_episode_cost": {"direction": "lower", "weight": 0.20},
        "cvar95_cost": {"direction": "lower", "weight": 0.20},
        "cr": {"direction": "higher", "weight": 0.10},
    },
    "auxiliary": {
        "v_dev": {"direction": "lower"},
        "max_v_drop_dev": {"direction": "lower"},
        "max_v_rise_dev": {"direction": "lower"},
        "no_destroy_rate": {"direction": "higher"},
        "constraint_satisfaction_rate": {"direction": "higher"},
        "avg_destroy": {"direction": "lower"},
    },
}


def _to_float(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _ok(
    baseline: Optional[float],
    offline: Optional[float],
    direction: str,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0
) -> bool:
    """
    Safe comparison:
    - If both are None: treat as pass (metric unavailable on both sides)
    - If baseline is None and offline is not None: skip strict failure, treat as pass
    - If baseline is not None and offline is None: treat as fail
    """
    baseline = _to_float(baseline, None)
    offline = _to_float(offline, None)

    if baseline is None and offline is None:
        return True
    if baseline is None and offline is not None:
        return True
    if baseline is not None and offline is None:
        return False

    if direction == "higher":
        return offline + abs_tol >= baseline * (1.0 - rel_tol)
    return offline <= baseline * (1.0 + rel_tol) + abs_tol


def _weighted_score(summary: Dict[str, float], spec: Dict[str, Dict]) -> float:
    score = 0.0
    for metric, cfg in spec.items():
        value = _to_float(summary.get(metric, None), None)
        if value is None:
            continue
        weight = float(cfg.get("weight", 0.0))
        if cfg["direction"] == "higher":
            score += weight * value
        else:
            score -= weight * value
    return score


def compare_with_relaxed_judgement(
    baseline_summary: Dict[str, float],
    offline_summary: Dict[str, float],
    return_drop_tolerance_ratio: float = 0.03,
    return_drop_tolerance_abs: float = 0.25,
    safety_rel_tol: float = 0.05,
    performance_aux_rel_tol: float = 0.10,
) -> Dict:
    perf_checks: List[Dict] = []
    safe_checks: List[Dict] = []
    aux_checks: List[Dict] = []

    baseline_return = _to_float(baseline_summary.get("avg_return", None), 0.0)
    offline_return = _to_float(offline_summary.get("avg_return", None), 0.0)

    # Key performance: allow small return drop.
    ret_ok = offline_return >= baseline_return - max(
        return_drop_tolerance_abs,
        abs(baseline_return) * return_drop_tolerance_ratio,
    )
    perf_checks.append({
        "metric": "avg_return",
        "baseline": baseline_return,
        "offline": offline_return,
        "better_or_equal": bool(ret_ok),
        "direction": "higher_is_better_with_small_drop_allowed",
    })

    for metric in ["cr", "pl", "avg_q_loss"]:
        direction = DEFAULT_KEY_METRICS["performance"][metric]["direction"]
        baseline_val = _to_float(baseline_summary.get(metric, None), None)
        offline_val = _to_float(offline_summary.get(metric, None), None)
        ok = _ok(
            baseline_val,
            offline_val,
            direction=direction,
            rel_tol=performance_aux_rel_tol,
            abs_tol=0.0,
        )
        perf_checks.append({
            "metric": metric,
            "baseline": baseline_val,
            "offline": offline_val,
            "better_or_equal": bool(ok),
            "direction": f"{direction}_is_better",
        })

    for metric, cfg in DEFAULT_KEY_METRICS["safety"].items():
        baseline_val = _to_float(baseline_summary.get(metric, None), None)
        offline_val = _to_float(offline_summary.get(metric, None), None)
        ok = _ok(
            baseline_val,
            offline_val,
            direction=cfg["direction"],
            rel_tol=safety_rel_tol,
            abs_tol=0.0,
        )
        safe_checks.append({
            "metric": metric,
            "baseline": baseline_val,
            "offline": offline_val,
            "better_or_equal": bool(ok),
            "direction": f"{cfg['direction']}_is_better",
        })

    for metric, cfg in DEFAULT_KEY_METRICS["auxiliary"].items():
        baseline_val = _to_float(baseline_summary.get(metric, None), None)
        offline_val = _to_float(offline_summary.get(metric, None), None)
        ok = _ok(
            baseline_val,
            offline_val,
            direction=cfg["direction"],
            rel_tol=0.10,
            abs_tol=0.0,
        )
        aux_checks.append({
            "metric": metric,
            "baseline": baseline_val,
            "offline": offline_val,
            "better_or_equal": bool(ok),
            "direction": f"{cfg['direction']}_is_better",
        })

    perf_score_baseline = _weighted_score(baseline_summary, DEFAULT_KEY_METRICS["performance"])
    perf_score_offline = _weighted_score(offline_summary, DEFAULT_KEY_METRICS["performance"])
    safe_score_baseline = _weighted_score(baseline_summary, DEFAULT_KEY_METRICS["safety"])
    safe_score_offline = _weighted_score(offline_summary, DEFAULT_KEY_METRICS["safety"])

    performance_not_lower_than_baseline = bool(
        ret_ok and perf_score_offline >= perf_score_baseline - 0.05 * max(1.0, abs(perf_score_baseline))
    )
    safety_not_worse_than_baseline_on_key_metrics = bool(all(x["better_or_equal"] for x in safe_checks))
    overall_claim_supported = bool(
        performance_not_lower_than_baseline and safety_not_worse_than_baseline_on_key_metrics
    )

    return {
        "performance_not_lower_than_baseline": performance_not_lower_than_baseline,
        "safety_not_worse_than_baseline_on_key_metrics": safety_not_worse_than_baseline_on_key_metrics,
        "overall_claim_supported": overall_claim_supported,
        "performance_score_baseline": perf_score_baseline,
        "performance_score_offline": perf_score_offline,
        "safety_score_baseline": safe_score_baseline,
        "safety_score_offline": safe_score_offline,
        "key_metric_spec": DEFAULT_KEY_METRICS,
        "performance_details": perf_checks,
        "safety_details": safe_checks,
        "auxiliary_details": aux_checks,
    }