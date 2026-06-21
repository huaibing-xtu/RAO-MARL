import os
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from agents.mlp_agent import MLPAgent as DeterministicMLPAgent
from agents.rnn_agent import RNNAgent as DeterministicRNNAgent
from critics.mlp_critic import MLPCritic
from critics.maac_critic import AttentionCritic


def build_mapdn_args(obs_dim: int, act_dim: int, n_agents: int, hid_size: int = 64,
                     layernorm: bool = True, hid_activation: str = "relu",
                     agent_type: str = "rnn", agent_id: bool = True,
                     shared_params: bool = True, continuous: bool = True,
                     attend_heads: int = 4, norm_in: bool = False):
    return SimpleNamespace(
        obs_size=obs_dim,
        action_dim=act_dim,
        agent_num=n_agents,
        hid_size=hid_size,
        layernorm=layernorm,
        hid_activation=hid_activation,
        agent_type=agent_type,
        agent_id=agent_id,
        shared_params=shared_params,
        continuous=continuous,
        attend_heads=attend_heads,
        norm_in=norm_in,
    )


class MAPDNSharedActor(nn.Module):
    """Use MAPDN policy backbone as offline actor.

    residual: action = clip(bc_action + delta)
    direct:   action = scaled(policy(obs))
    """
    def __init__(self, obs_dim: int, act_dim: int, n_agents: int,
                 action_low: float, action_high: float, phi: float = 0.05,
                 model_name: str = "matd3", mode: str = "residual",
                 hid_size: int = 64, agent_type: str = "rnn"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.phi = float(phi)
        self.mode = mode
        self.model_name = model_name
        self.args = build_mapdn_args(
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_agents=n_agents,
            hid_size=hid_size,
            agent_type=agent_type,
        )
        input_shape = obs_dim + n_agents
        agent_cls = DeterministicRNNAgent if agent_type == "rnn" else DeterministicMLPAgent
        self.policy = agent_cls(input_shape, self.args)

    def _augment_obs(self, obs: torch.Tensor) -> torch.Tensor:
        total = obs.shape[0]
        agent_ids = torch.eye(self.n_agents, device=obs.device).repeat(total // self.n_agents, 1)
        return torch.cat([obs, agent_ids], dim=-1)

    def forward(self, obs: torch.Tensor, bc_action: torch.Tensor) -> torch.Tensor:
        aug_obs = self._augment_obs(obs)
        if self.args.agent_type == "rnn":
            hid = torch.zeros(obs.shape[0], self.args.hid_size, device=obs.device, dtype=obs.dtype)
            raw, _, _ = self.policy(aug_obs, hid)
        else:
            raw, _, _ = self.policy(aug_obs, None)
        raw = torch.tanh(raw)
        if self.mode == "direct":
            action = self.action_low + 0.5 * (raw + 1.0) * (self.action_high - self.action_low)
            return torch.clamp(action, self.action_low, self.action_high)
        delta = self.phi * (self.action_high - self.action_low) * raw
        action = bc_action + delta
        return torch.clamp(action, self.action_low, self.action_high)

    def load_from_mapdn_checkpoint(self, ckpt_path: str) -> None:
        if not ckpt_path or not os.path.exists(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        prefix = "policy_dicts.0."
        actor_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        self.policy.load_state_dict(actor_sd, strict=False)


class MAPDNCritic(nn.Module):
    """Critic variants reusing MAPDN-side building blocks.

    critic_name:
      - central: original offline_safe central critic
      - maddpg/matd3: centralized joint critic similar to MAPDN MADDPG/MATD3 style
      - iddpg: independent per-agent critic averaged over agents
      - maac: attention critic averaged over agents
    """
    def __init__(self, state_dim: int, obs_dim: int, n_agents: int, act_dim: int,
                 critic_name: str = "central", hid_size: int = 64, agent_type: str = "rnn"):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.critic_name = critic_name
        self.args = build_mapdn_args(
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_agents=n_agents,
            hid_size=hid_size,
            agent_type=agent_type,
        )
        if critic_name == "central":
            self.net = nn.Sequential(
                nn.Linear(state_dim + n_agents * act_dim, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1)
            )
        elif critic_name in {"maddpg", "matd3"}:
            input_shape = (obs_dim + act_dim) * n_agents + n_agents
            self.net = MLPCritic(input_shape, 1, self.args)
        elif critic_name == "iddpg":
            input_shape = obs_dim + act_dim + n_agents
            self.net = MLPCritic(input_shape, 1, self.args)
        elif critic_name == "maac":
            self.net = AttentionCritic(self.args)
        else:
            raise ValueError(f"Unknown critic_name: {critic_name}")

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor, obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz = joint_action.shape[0]
        if self.critic_name == "central":
            x = torch.cat([state, joint_action.reshape(bsz, -1)], dim=-1)
            return self.net(x)
        if obs is None:
            raise ValueError(f"critic_name={self.critic_name} requires obs input")
        if self.critic_name in {"maddpg", "matd3"}:
            obs_repeat = obs.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
            obs_flat = obs_repeat.reshape(bsz * self.n_agents, -1)
            act_repeat = joint_action.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
            act_flat = act_repeat.reshape(bsz * self.n_agents, -1)
            agent_ids = torch.eye(self.n_agents, device=obs.device).unsqueeze(0).repeat(bsz, 1, 1).reshape(bsz * self.n_agents, -1)
            x = torch.cat([obs_flat, act_flat, agent_ids], dim=-1)
            q, _ = self.net(x, None)
            return q.reshape(bsz, self.n_agents, 1).mean(dim=1)
        if self.critic_name == "iddpg":
            agent_ids = torch.eye(self.n_agents, device=obs.device).unsqueeze(0).repeat(bsz, 1, 1)
            x = torch.cat([obs, joint_action, agent_ids], dim=-1).reshape(bsz * self.n_agents, -1)
            q, _ = self.net(x, None)
            return q.reshape(bsz, self.n_agents, 1).mean(dim=1)
        obs_chunks = [x.squeeze(1) for x in torch.chunk(obs, self.n_agents, dim=1)]
        act_chunks = [x.squeeze(1) for x in torch.chunk(joint_action, self.n_agents, dim=1)]
        sa = torch.cat([obs, joint_action], dim=-1)
        sa_chunks = [x.squeeze(1) for x in torch.chunk(sa, self.n_agents, dim=1)]
        rets = self.net((obs_chunks, act_chunks, sa_chunks))
        qs = torch.cat([ret[0] for ret in rets], dim=1)
        return qs.mean(dim=1, keepdim=True)
