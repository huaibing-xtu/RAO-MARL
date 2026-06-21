import os
import glob
import json
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class MAPDNTransitionDataset(Dataset):
    """
    读取 offline_safe/data/... 下的 ep_*.npz，并把 episode 级轨迹展平为 transition。

    每个 npz 需要至少包含：
        obs, state, actions, rewards, costs, next_obs, next_state, dones

    输出 sample 的形状：
        obs:        [N, obs_dim]
        state:      [state_dim]
        actions:    [N, act_dim]
        rewards:    [1]
        costs:      [1]
        next_obs:   [N, obs_dim]
        next_state: [state_dim]
        dones:      [1]
    """

    REQUIRED_KEYS = (
        "obs",
        "state",
        "actions",
        "rewards",
        "costs",
        "next_obs",
        "next_state",
        "dones",
    )

    def __init__(
        self,
        root_dirs: Sequence[str],
        normalize_obs: bool = True,
        normalize_state: bool = True,
        cache_in_memory: bool = True,
    ):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = list(root_dirs)
        self.normalize_obs = normalize_obs
        self.normalize_state = normalize_state
        self.cache_in_memory = cache_in_memory

        self.file_list = self._collect_files(self.root_dirs)
        if len(self.file_list) == 0:
            raise ValueError(f"No ep_*.npz files found in: {self.root_dirs}")

        self.data = self._load_all_files(self.file_list) if cache_in_memory else None
        self._build_index()
        self._compute_stats()

    @staticmethod
    def _collect_files(root_dirs: Sequence[str]) -> List[str]:
        file_list: List[str] = []
        for root in root_dirs:
            file_list.extend(sorted(glob.glob(os.path.join(root, "ep_*.npz"))))
        return file_list

    def _load_all_files(self, file_list: Sequence[str]) -> List[Dict[str, np.ndarray]]:
        data = []
        for fp in file_list:
            with np.load(fp) as d:
                item = {k: d[k].astype(np.float32) for k in d.files}
            self._validate_episode(item, fp)
            data.append(item)
        return data

    def _load_single_file(self, fp: str) -> Dict[str, np.ndarray]:
        with np.load(fp) as d:
            item = {k: d[k].astype(np.float32) for k in d.files}
        self._validate_episode(item, fp)
        return item

    def _validate_episode(self, item: Dict[str, np.ndarray], fp: str) -> None:
        for key in self.REQUIRED_KEYS:
            if key not in item:
                raise KeyError(f"{fp} missing required key: {key}")

        length = item["obs"].shape[0]
        for key in self.REQUIRED_KEYS:
            if item[key].shape[0] != length:
                raise ValueError(
                    f"{fp} has inconsistent length: {key} length {item[key].shape[0]} != obs length {length}"
                )

    def _build_index(self) -> None:
        self.index = []
        self.episode_lengths = []
        for epi, fp in enumerate(self.file_list):
            item = self.data[epi] if self.cache_in_memory else self._load_single_file(fp)
            length = item["obs"].shape[0]
            self.episode_lengths.append(length)
            for t in range(length):
                self.index.append((epi, t))

    def _compute_stats(self) -> None:
        # 计算 obs / state 归一化统计
        obs_sum = None
        obs_sq_sum = None
        obs_count = 0

        state_sum = None
        state_sq_sum = None
        state_count = 0

        for epi, fp in enumerate(self.file_list):
            item = self.data[epi] if self.cache_in_memory else self._load_single_file(fp)

            obs = item["obs"]          # [T, N, O]
            state = item["state"]      # [T, S]

            if obs_sum is None:
                obs_sum = obs.sum(axis=(0, 1), keepdims=True)
                obs_sq_sum = (obs ** 2).sum(axis=(0, 1), keepdims=True)
            else:
                obs_sum += obs.sum(axis=(0, 1), keepdims=True)
                obs_sq_sum += (obs ** 2).sum(axis=(0, 1), keepdims=True)
            obs_count += obs.shape[0] * obs.shape[1]

            if state_sum is None:
                state_sum = state.sum(axis=0, keepdims=True)
                state_sq_sum = (state ** 2).sum(axis=0, keepdims=True)
            else:
                state_sum += state.sum(axis=0, keepdims=True)
                state_sq_sum += (state ** 2).sum(axis=0, keepdims=True)
            state_count += state.shape[0]

        self.obs_mean = (obs_sum / max(obs_count, 1)).astype(np.float32)
        self.obs_var = (obs_sq_sum / max(obs_count, 1) - self.obs_mean ** 2).astype(np.float32)
        self.obs_std = np.sqrt(np.maximum(self.obs_var, 1e-8)).astype(np.float32)

        self.state_mean = (state_sum / max(state_count, 1)).astype(np.float32)
        self.state_var = (state_sq_sum / max(state_count, 1) - self.state_mean ** 2).astype(np.float32)
        self.state_std = np.sqrt(np.maximum(self.state_var, 1e-8)).astype(np.float32)

        # 数据维度信息
        first = self.data[0] if self.cache_in_memory else self._load_single_file(self.file_list[0])
        self.n_agents = int(first["obs"].shape[1])
        self.obs_dim = int(first["obs"].shape[2])
        self.state_dim = int(first["state"].shape[1])
        self.act_dim = int(first["actions"].shape[2])

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
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi, t = self.index[idx]
        item = self.data[epi] if self.cache_in_memory else self._load_single_file(self.file_list[epi])

        obs = item["obs"][t]               # [N, O]
        state = item["state"][t]           # [S]
        actions = item["actions"][t]       # [N, A]
        rewards = item["rewards"][t]       # [1]
        costs = item["costs"][t]           # [1]
        next_obs = item["next_obs"][t]     # [N, O]
        next_state = item["next_state"][t] # [S]
        dones = item["dones"][t]           # [1]

        if self.normalize_obs:
            obs = (obs - self.obs_mean.squeeze(0)) / (self.obs_std.squeeze(0) + 1e-8)
            next_obs = (next_obs - self.obs_mean.squeeze(0)) / (self.obs_std.squeeze(0) + 1e-8)

        if self.normalize_state:
            state = (state - self.state_mean.squeeze(0)) / (self.state_std.squeeze(0) + 1e-8)
            next_state = (next_state - self.state_mean.squeeze(0)) / (self.state_std.squeeze(0) + 1e-8)

        sample = {
            "obs": torch.from_numpy(obs.astype(np.float32)),
            "state": torch.from_numpy(state.astype(np.float32)),
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "rewards": torch.from_numpy(rewards.astype(np.float32)),
            "costs": torch.from_numpy(costs.astype(np.float32)),
            "next_obs": torch.from_numpy(next_obs.astype(np.float32)),
            "next_state": torch.from_numpy(next_state.astype(np.float32)),
            "dones": torch.from_numpy(dones.astype(np.float32)),
        }
        return sample
