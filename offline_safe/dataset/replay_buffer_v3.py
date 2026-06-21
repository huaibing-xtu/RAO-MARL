import os
import glob
import json
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class MAPDNTransitionDataset(Dataset):
    REQUIRED_KEYS = ("obs", "state", "actions", "rewards", "costs", "next_obs", "next_state", "dones")

    def __init__(
        self,
        root_dirs: Sequence[str],
        normalize_obs: bool = True,
        normalize_state: bool = True,
        cache_in_memory: bool = True,
        gamma: float = 0.99,
        budget_quantile: float = 0.50,
        unsafe_weight_coef: float = 1.50,
        actor_cost_quantile: float = 0.60,
        actor_disc_cost_quantile: float = 0.60,
        actor_return_quantile: float = 0.30,
        actor_min_keep_ratio: float = 0.35,
        actor_good_weight: float = 1.0,
        actor_medium_weight: float = 0.35,
        actor_bad_weight: float = 0.0,
    ):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = list(root_dirs)
        self.normalize_obs = normalize_obs
        self.normalize_state = normalize_state
        self.cache_in_memory = cache_in_memory
        self.gamma = float(gamma)
        self.budget_quantile = float(budget_quantile)
        self.unsafe_weight_coef = float(unsafe_weight_coef)

        self.actor_cost_quantile = float(actor_cost_quantile)
        self.actor_disc_cost_quantile = float(actor_disc_cost_quantile)
        self.actor_return_quantile = float(actor_return_quantile)
        self.actor_min_keep_ratio = float(actor_min_keep_ratio)
        self.actor_good_weight = float(actor_good_weight)
        self.actor_medium_weight = float(actor_medium_weight)
        self.actor_bad_weight = float(actor_bad_weight)

        self.file_list = self._collect_files(self.root_dirs)
        if not self.file_list:
            raise ValueError(f"No ep_*.npz files found in: {self.root_dirs}")

        self.data = self._load_all_files(self.file_list) if cache_in_memory else None
        if cache_in_memory:
            self._build_index()
            self._compute_stats()
            self._compute_budget_stats()
            self._compute_actor_episode_policy()
        else:
            self._init_from_disk_single_pass()

    @staticmethod
    def _collect_files(root_dirs: Sequence[str]) -> List[str]:
        out: List[str] = []
        for root in root_dirs:
            out.extend(sorted(glob.glob(os.path.join(root, "ep_*.npz"))))
        return out

    def _validate_episode(self, item: Dict[str, np.ndarray], fp: str) -> None:
        for k in self.REQUIRED_KEYS:
            if k not in item:
                raise KeyError(f"{fp} missing required key: {k}")
        t = item["obs"].shape[0]
        for k in self.REQUIRED_KEYS:
            if item[k].shape[0] != t:
                raise ValueError(f"{fp} inconsistent length: {k}={item[k].shape[0]} vs obs={t}")

    def _augment_episode_fields(self, item: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        rewards = np.asarray(item["rewards"], dtype=np.float32).reshape(-1)
        costs = np.asarray(item["costs"], dtype=np.float32).reshape(-1)
        t = len(rewards)

        reward_to_go = np.zeros((t, 1), dtype=np.float32)
        cost_to_go = np.zeros((t, 1), dtype=np.float32)
        rr = 0.0
        rc = 0.0
        for i in reversed(range(t)):
            rr = float(rewards[i]) + self.gamma * rr
            rc = float(costs[i]) + self.gamma * rc
            reward_to_go[i, 0] = rr
            cost_to_go[i, 0] = rc

        item["reward_to_go"] = reward_to_go
        item["cost_to_go"] = cost_to_go
        item["timestep"] = np.arange(t, dtype=np.float32).reshape(-1, 1)
        item["remaining_steps"] = (t - 1 - np.arange(t, dtype=np.float32)).reshape(-1, 1)
        item["episode_return"] = np.asarray([rewards.sum()], dtype=np.float32)
        item["episode_cost"] = np.asarray([costs.sum()], dtype=np.float32)
        item["episode_discounted_cost"] = np.asarray([cost_to_go[0, 0]], dtype=np.float32)
        return item

    def _load_all_files(self, file_list: Sequence[str]) -> List[Dict[str, np.ndarray]]:
        data: List[Dict[str, np.ndarray]] = []
        for fp in file_list:
            with np.load(fp) as d:
                item = {k: d[k].astype(np.float32) for k in d.files}
            self._validate_episode(item, fp)
            data.append(self._augment_episode_fields(item))
        return data

    def _load_single_file(self, fp: str) -> Dict[str, np.ndarray]:
        with np.load(fp, mmap_mode='r') as d:
            item = {}
            for k in d.files:
                arr = d[k]
                item[k] = arr if arr.dtype == np.float32 else arr.astype(np.float32)
        self._validate_episode(item, fp)
        return self._augment_episode_fields(item)

    def _build_index(self) -> None:
        self.index: List[tuple[int, int]] = []
        self.episode_lengths: List[int] = []
        for epi, fp in enumerate(self.file_list):
            item = self.data[epi] if self.cache_in_memory else self._load_single_file(fp)
            t = item["obs"].shape[0]
            self.episode_lengths.append(t)
            for step in range(t):
                self.index.append((epi, step))

    def _compute_stats(self) -> None:
        obs_sum = obs_sq_sum = state_sum = state_sq_sum = None
        obs_count = state_count = 0
        for epi, fp in enumerate(self.file_list):
            item = self.data[epi] if self.cache_in_memory else self._load_single_file(fp)
            obs = item["obs"]
            state = item["state"]
            if obs_sum is None:
                obs_sum = obs.sum(axis=(0, 1), keepdims=True)
                obs_sq_sum = (obs ** 2).sum(axis=(0, 1), keepdims=True)
                state_sum = state.sum(axis=0, keepdims=True)
                state_sq_sum = (state ** 2).sum(axis=0, keepdims=True)
            else:
                obs_sum += obs.sum(axis=(0, 1), keepdims=True)
                obs_sq_sum += (obs ** 2).sum(axis=(0, 1), keepdims=True)
                state_sum += state.sum(axis=0, keepdims=True)
                state_sq_sum += (state ** 2).sum(axis=0, keepdims=True)
            obs_count += obs.shape[0] * obs.shape[1]
            state_count += state.shape[0]

        self.obs_mean = (obs_sum / max(obs_count, 1)).astype(np.float32)
        self.obs_var = (obs_sq_sum / max(obs_count, 1) - self.obs_mean ** 2).astype(np.float32)
        self.obs_std = np.sqrt(np.maximum(self.obs_var, 1e-8)).astype(np.float32)

        self.state_mean = (state_sum / max(state_count, 1)).astype(np.float32)
        self.state_var = (state_sq_sum / max(state_count, 1) - self.state_mean ** 2).astype(np.float32)
        self.state_std = np.sqrt(np.maximum(self.state_var, 1e-8)).astype(np.float32)

        first = self.data[0] if self.cache_in_memory else self._load_single_file(self.file_list[0])
        self.n_agents = int(first["obs"].shape[1])
        self.obs_dim = int(first["obs"].shape[2])
        self.state_dim = int(first["state"].shape[1])
        self.act_dim = int(first["actions"].shape[2])

    def _compute_budget_stats(self) -> None:
        max_len = max(self.episode_lengths) if self.episode_lengths else 0
        step_values: List[List[float]] = [[] for _ in range(max_len)]
        for epi, fp in enumerate(self.file_list):
            item = self.data[epi] if self.cache_in_memory else self._load_single_file(fp)
            ctg = item["cost_to_go"].reshape(-1)
            for t in range(len(ctg)):
                step_values[t].append(float(ctg[t]))
        budgets: List[float] = []
        for t in range(max_len):
            vals = step_values[t]
            budgets.append(float(np.quantile(vals, self.budget_quantile)) if vals else (budgets[-1] if budgets else 0.0))
        self.timestep_cost_budgets = np.asarray(budgets, dtype=np.float32)

    def _compute_actor_episode_policy(self) -> None:
        episode_returns = []
        episode_costs = []
        episode_disc_costs = []
        for epi, fp in enumerate(self.file_list):
            item = self.data[epi] if self.cache_in_memory else self._load_single_file(fp)
            episode_returns.append(float(item["episode_return"][0]))
            episode_costs.append(float(item["episode_cost"][0]))
            episode_disc_costs.append(float(item["episode_discounted_cost"][0]))

        returns = np.asarray(episode_returns, dtype=np.float32)
        costs = np.asarray(episode_costs, dtype=np.float32)
        disc_costs = np.asarray(episode_disc_costs, dtype=np.float32)

        cost_thr = float(np.quantile(costs, self.actor_cost_quantile))
        disc_thr = float(np.quantile(disc_costs, self.actor_disc_cost_quantile))
        ret_thr = float(np.quantile(returns, self.actor_return_quantile))

        keep = (costs <= cost_thr) & (disc_costs <= disc_thr) & (returns >= ret_thr)
        if keep.mean() < self.actor_min_keep_ratio:
            cost_rank = np.argsort(np.argsort(costs)).astype(np.float32) / max(len(costs) - 1, 1)
            disc_rank = np.argsort(np.argsort(disc_costs)).astype(np.float32) / max(len(costs) - 1, 1)
            ret_rank = np.argsort(np.argsort(returns)).astype(np.float32) / max(len(costs) - 1, 1)
            score = 0.45 * ret_rank - 0.30 * cost_rank - 0.25 * disc_rank
            topk = max(1, int(round(len(score) * self.actor_min_keep_ratio)))
            order = np.argsort(-score)
            keep = np.zeros_like(costs, dtype=bool)
            keep[order[:topk]] = True
        medium = (~keep) & (costs <= np.quantile(costs, 0.85)) & (disc_costs <= np.quantile(disc_costs, 0.85))

        actor_weight = np.full_like(costs, self.actor_bad_weight, dtype=np.float32)
        actor_weight[medium] = self.actor_medium_weight
        actor_weight[keep] = self.actor_good_weight

        self.actor_episode_keep = keep.astype(np.float32)
        self.actor_episode_weight = actor_weight.astype(np.float32)
        self.actor_policy_meta = {
            "actor_cost_quantile": self.actor_cost_quantile,
            "actor_disc_cost_quantile": self.actor_disc_cost_quantile,
            "actor_return_quantile": self.actor_return_quantile,
            "actor_min_keep_ratio": self.actor_min_keep_ratio,
            "actor_keep_ratio": float(self.actor_episode_keep.mean()),
            "cost_threshold": cost_thr,
            "discounted_cost_threshold": disc_thr,
            "return_threshold": ret_thr,
            "actor_good_weight": self.actor_good_weight,
            "actor_medium_weight": self.actor_medium_weight,
            "actor_bad_weight": self.actor_bad_weight,
        }

    def _init_from_disk_single_pass(self) -> None:
        """Single-pass initialization that avoids repeated file loads."""
        import gc
        self.index: List[tuple] = []
        self.episode_lengths: List[int] = []
        obs_sum = obs_sq_sum = state_sum = state_sq_sum = None
        obs_count = state_count = 0
        episode_returns = []
        episode_costs = []
        episode_disc_costs = []
        step_values_dict: Dict[int, List[float]] = {}

        for epi, fp in enumerate(self.file_list):
            item = self._load_single_file(fp)
            t = item["obs"].shape[0]
            self.episode_lengths.append(t)
            for step in range(t):
                self.index.append((epi, step))

            obs = item["obs"]; state = item["state"]
            if obs_sum is None:
                obs_sum = obs.sum(axis=(0, 1), keepdims=True)
                obs_sq_sum = (obs ** 2).sum(axis=(0, 1), keepdims=True)
                state_sum = state.sum(axis=0, keepdims=True)
                state_sq_sum = (state ** 2).sum(axis=0, keepdims=True)
            else:
                obs_sum += obs.sum(axis=(0, 1), keepdims=True)
                obs_sq_sum += (obs ** 2).sum(axis=(0, 1), keepdims=True)
                state_sum += state.sum(axis=0, keepdims=True)
                state_sq_sum += (state ** 2).sum(axis=0, keepdims=True)
            obs_count += obs.shape[0] * obs.shape[1]
            state_count += state.shape[0]

            ctg = item["cost_to_go"].reshape(-1)
            for step_t in range(len(ctg)):
                if step_t not in step_values_dict:
                    step_values_dict[step_t] = []
                step_values_dict[step_t].append(float(ctg[step_t]))

            episode_returns.append(float(item["episode_return"][0]))
            episode_costs.append(float(item["episode_cost"][0]))
            episode_disc_costs.append(float(item["episode_discounted_cost"][0]))

            if (epi + 1) % 200 == 0:
                gc.collect()

        self.obs_mean = (obs_sum / max(obs_count, 1)).astype(np.float32)
        self.obs_var = (obs_sq_sum / max(obs_count, 1) - self.obs_mean ** 2).astype(np.float32)
        self.obs_std = np.sqrt(np.maximum(self.obs_var, 1e-8)).astype(np.float32)
        self.state_mean = (state_sum / max(state_count, 1)).astype(np.float32)
        self.state_var = (state_sq_sum / max(state_count, 1) - self.state_mean ** 2).astype(np.float32)
        self.state_std = np.sqrt(np.maximum(self.state_var, 1e-8)).astype(np.float32)

        first = self._load_single_file(self.file_list[0])
        self.n_agents = int(first["obs"].shape[1])
        self.obs_dim = int(first["obs"].shape[2])
        self.state_dim = int(first["state"].shape[1])
        self.act_dim = int(first["actions"].shape[2])

        max_len = max(self.episode_lengths) if self.episode_lengths else 0
        budgets: List[float] = []
        for t in range(max_len):
            vals = step_values_dict.get(t, [])
            budgets.append(float(np.quantile(vals, self.budget_quantile)) if vals else (budgets[-1] if budgets else 0.0))
        self.timestep_cost_budgets = np.asarray(budgets, dtype=np.float32)
        del step_values_dict; gc.collect()

        returns = np.asarray(episode_returns, dtype=np.float32)
        costs = np.asarray(episode_costs, dtype=np.float32)
        disc_costs = np.asarray(episode_disc_costs, dtype=np.float32)
        cost_thr = float(np.quantile(costs, self.actor_cost_quantile))
        disc_thr = float(np.quantile(disc_costs, self.actor_disc_cost_quantile))
        ret_thr = float(np.quantile(returns, self.actor_return_quantile))
        keep = (costs <= cost_thr) & (disc_costs <= disc_thr) & (returns >= ret_thr)
        if keep.mean() < self.actor_min_keep_ratio:
            cost_rank = np.argsort(np.argsort(costs)).astype(np.float32) / max(len(costs) - 1, 1)
            disc_rank = np.argsort(np.argsort(disc_costs)).astype(np.float32) / max(len(costs) - 1, 1)
            ret_rank = np.argsort(np.argsort(returns)).astype(np.float32) / max(len(costs) - 1, 1)
            score = 0.45 * ret_rank - 0.30 * cost_rank - 0.25 * disc_rank
            topk = max(1, int(round(len(score) * self.actor_min_keep_ratio)))
            order = np.argsort(-score)
            keep = np.zeros_like(costs, dtype=bool)
            keep[order[:topk]] = True
        medium = (~keep) & (costs <= np.quantile(costs, 0.85)) & (disc_costs <= np.quantile(disc_costs, 0.85))
        actor_weight = np.full_like(costs, self.actor_bad_weight, dtype=np.float32)
        actor_weight[medium] = self.actor_medium_weight
        actor_weight[keep] = self.actor_good_weight
        self.actor_episode_keep = keep.astype(np.float32)
        self.actor_episode_weight = actor_weight.astype(np.float32)
        self.actor_policy_meta = {
            "actor_cost_quantile": self.actor_cost_quantile,
            "actor_disc_cost_quantile": self.actor_disc_cost_quantile,
            "actor_return_quantile": self.actor_return_quantile,
            "actor_min_keep_ratio": self.actor_min_keep_ratio,
            "actor_keep_ratio": float(self.actor_episode_keep.mean()),
            "cost_threshold": cost_thr,
            "discounted_cost_threshold": disc_thr,
            "return_threshold": ret_thr,
            "actor_good_weight": self.actor_good_weight,
            "actor_medium_weight": self.actor_medium_weight,
            "actor_bad_weight": self.actor_bad_weight,
        }

    def save_stats(self, path: str) -> None:
        obj = {
            "num_episodes": len(self.file_list),
            "num_transitions": len(self.index),
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "act_dim": self.act_dim,
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "gamma": self.gamma,
            "budget_quantile": self.budget_quantile,
            "timestep_cost_budgets": self.timestep_cost_budgets.tolist(),
            "actor_policy_meta": self.actor_policy_meta,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi, t = self.index[idx]
        item = self.data[epi] if self.cache_in_memory else self._load_single_file(self.file_list[epi])

        obs = item["obs"][t]
        state = item["state"][t]
        actions = item["actions"][t]
        rewards = item["rewards"][t]
        costs = item["costs"][t]
        next_obs = item["next_obs"][t]
        next_state = item["next_state"][t]
        dones = item["dones"][t]
        reward_to_go = item["reward_to_go"][t]
        cost_to_go = item["cost_to_go"][t]
        timestep = item["timestep"][t]
        remaining = item["remaining_steps"][t]

        if self.normalize_obs:
            obs = (obs - self.obs_mean.squeeze(0)) / (self.obs_std.squeeze(0) + 1e-8)
            next_obs = (next_obs - self.obs_mean.squeeze(0)) / (self.obs_std.squeeze(0) + 1e-8)
        if self.normalize_state:
            state = (state - self.state_mean.squeeze(0)) / (self.state_std.squeeze(0) + 1e-8)
            next_state = (next_state - self.state_mean.squeeze(0)) / (self.state_std.squeeze(0) + 1e-8)

        i = min(int(t), max(0, len(self.timestep_cost_budgets) - 1))
        j = min(int(t + 1), max(0, len(self.timestep_cost_budgets) - 1))
        state_budget = np.asarray([self.timestep_cost_budgets[i]], dtype=np.float32)
        next_state_budget = np.asarray([self.timestep_cost_budgets[j]], dtype=np.float32)
        ratio = float(cost_to_go.reshape(-1)[0]) / max(float(state_budget[0]), 1e-6)
        unsafe_weight = np.asarray([1.0 + self.unsafe_weight_coef * max(0.0, ratio - 1.0)], dtype=np.float32)

        actor_mask = np.asarray([self.actor_episode_keep[epi]], dtype=np.float32)
        actor_weight = np.asarray([self.actor_episode_weight[epi]], dtype=np.float32)
        vae_weight = np.asarray([self.actor_episode_weight[epi]], dtype=np.float32)

        cast = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32))
        return {
            "obs": cast(obs),
            "state": cast(state),
            "actions": cast(actions),
            "rewards": cast(rewards),
            "costs": cast(costs),
            "next_obs": cast(next_obs),
            "next_state": cast(next_state),
            "dones": cast(dones),
            "reward_to_go": cast(reward_to_go),
            "cost_to_go": cast(cost_to_go),
            "timestep": cast(timestep),
            "remaining_steps": cast(remaining),
            "state_budget": cast(state_budget),
            "next_state_budget": cast(next_state_budget),
            "unsafe_weight": cast(unsafe_weight),
            "actor_mask": cast(actor_mask),
            "actor_weight": cast(actor_weight),
            "vae_weight": cast(vae_weight),
        }
