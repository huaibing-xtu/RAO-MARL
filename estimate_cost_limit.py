# estimate_cost_limit.py
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

def load_episode_costs(data_root: Union[str, Path], gamma: float = 0.99) -> Tuple[List[float], List[float]]:
    """
    加载数据集，返回每个 episode 的 total_cost 和 discounted_cost。
    支持两种格式：
      1. 单个 dataset.npz 文件（所有 transition 平铺）
      2. 目录下包含多个 ep_*.npz 文件（每个文件是一个 episode）
    """
    data_root = Path(data_root)
    total_costs = []
    discounted_costs = []

    # 尝试读取单个 npz 文件
    npz_file = data_root / "dataset.npz"
    if npz_file.exists():
        data = np.load(npz_file)
        # 需要知道 episode 边界。假设数据集中有 day_index 字段，不同 day_index 对应不同 episode
        if "day_index" in data:
            day_indices = data["day_index"]
            unique_days = np.unique(day_indices)
            for day in unique_days:
                mask = day_indices == day
                costs = data["cost"][mask]  # 注意字段名可能是 "cost" 或 "costs"
                if len(costs) == 0:
                    continue
                total_costs.append(np.sum(costs))
                disc = 0.0
                for t, c in enumerate(costs):
                    disc += (gamma ** t) * float(c)
                discounted_costs.append(disc)
        else:
            # 如果没有 day_index，则假设整个文件是一个 episode
            costs = data["cost"]  # shape: (n_transitions,)
            total_costs.append(np.sum(costs))
            disc = 0.0
            for t, c in enumerate(costs):
                disc += (gamma ** t) * float(c)
            discounted_costs.append(disc)
    else:
        # 尝试读取多个 ep_*.npz 文件
        ep_files = sorted(data_root.glob("ep_*.npz"))
        if not ep_files:
            raise FileNotFoundError(f"No dataset found in {data_root}")
        for ep_file in ep_files:
            with np.load(ep_file) as data:
                costs = data["costs"].flatten()  # 每个 episode 的成本序列
                total_costs.append(np.sum(costs))
                disc = 0.0
                for t, c in enumerate(costs):
                    disc += (gamma ** t) * float(c)
                discounted_costs.append(disc)

    return total_costs, discounted_costs


def suggest_cost_limit(total_costs: List[float],
                       method: str = "quantile",
                       quantile: float = 0.8,
                       baseline_json: str = None,
                       manual_value: float = None) -> float:
    """
    根据指定方法确定 cost_limit。
    """
    if method == "manual":
        if manual_value is None:
            raise ValueError("--manual-value must be provided when method=manual")
        return manual_value

    if method == "baseline":
        if baseline_json is None:
            raise ValueError("--baseline-json must be provided when method=baseline")
        with open(baseline_json, "r") as f:
            data = json.load(f)
        # 基线 JSON 中可能包含 summary 字段
        if "summary" in data:
            summary = data["summary"]
        else:
            summary = data
        avg_cost = summary.get("avg_cost")
        if avg_cost is None:
            raise KeyError("avg_cost not found in baseline JSON")
        return float(avg_cost)

    if method == "quantile":
        if not total_costs:
            raise ValueError("No episode costs to compute quantile")
        return float(np.quantile(total_costs, quantile))

    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Estimate cost_limit from offline dataset.")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to dataset (directory containing dataset.npz or ep_*.npz)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for discounted cost calculation")
    parser.add_argument("--method", type=str, default="quantile",
                        choices=["quantile", "baseline", "manual"],
                        help="Method to determine cost_limit")
    parser.add_argument("--quantile", type=float, default=0.8,
                        help="Quantile for method='quantile' (0~1)")
    parser.add_argument("--baseline-json", type=str, default=None,
                        help="Path to baseline evaluation JSON for method='baseline'")
    parser.add_argument("--manual-value", type=float, default=None,
                        help="Manual cost_limit for method='manual'")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save suggested cost_limit to JSON file")
    args = parser.parse_args()

    # 加载成本数据
    total_costs, discounted_costs = load_episode_costs(args.data_root, gamma=args.gamma)
    print(f"Loaded {len(total_costs)} episodes.")
    print(f"Total cost statistics:")
    print(f"  min: {np.min(total_costs):.4f}")
    print(f"  median: {np.median(total_costs):.4f}")
    print(f"  mean: {np.mean(total_costs):.4f}")
    print(f"  80% quantile: {np.quantile(total_costs, 0.8):.4f}")
    print(f"  90% quantile: {np.quantile(total_costs, 0.9):.4f}")
    print(f"  max: {np.max(total_costs):.4f}")

    # 确定 cost_limit
    cost_limit = suggest_cost_limit(
        total_costs=total_costs,
        method=args.method,
        quantile=args.quantile,
        baseline_json=args.baseline_json,
        manual_value=args.manual_value,
    )
    print(f"\nSuggested cost_limit: {cost_limit:.4f} (method={args.method})")

    # 可选保存到文件
    if args.output_json:
        output = {
            "cost_limit": cost_limit,
            "method": args.method,
            "total_costs_stats": {
                "min": float(np.min(total_costs)),
                "median": float(np.median(total_costs)),
                "mean": float(np.mean(total_costs)),
                "quantile_80": float(np.quantile(total_costs, 0.8)),
                "quantile_90": float(np.quantile(total_costs, 0.9)),
                "max": float(np.max(total_costs)),
            }
        }
        if args.method == "quantile":
            output["quantile"] = args.quantile
        elif args.method == "baseline":
            output["baseline_json"] = args.baseline_json
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved suggestion to {args.output_json}")


if __name__ == "__main__":
    main()