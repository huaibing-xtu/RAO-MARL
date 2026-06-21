import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_safe.models.registry import build_behavior_model, build_actor_model, build_critic_model


class MABCQRetainLag:
    """
    A stronger offline-safe variant for MAPDN.

    Goals:
    1) Keep the strong safety gains of BCQ-Lag.
    2) Better preserve baseline performance with stronger behavior retention.
    3) Reduce offline overestimation using conservative critic regularization.

    Main changes over MABCQLag:
    - reward / cost critic CQL regularization
    - actor warmup imitation stage
    - actor loss with adaptive BC retention term
    - optional action L2 regularization for smoother actions
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        act_dim: int,
        n_agents: int,
        action_low: float,
        action_high: float,
        device: str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.005,
        phi: float = 0.05,
        latent_dim: int = 16,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        vae_lr: float = 3e-4,
        lag_lr: float = 1e-4,
        cost_limit: float = 5.0,
        bc_coef: float = 0.2,
        bc_coef_end: float = 0.05,
        action_l2_coef: float = 1e-4,
        cql_alpha_reward: float = 0.05,
        cql_alpha_cost: float = 0.05,
        warmup_actor_steps: int = 1000,
        actor_model_name: str = "mlp",
        critic_model_name: str = "mlp",
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(512, 512),
        critic_attend_heads: int = 4,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.cost_limit = float(cost_limit)
        self.bc_coef = float(bc_coef)
        self.bc_coef_end = float(bc_coef_end)
        self.action_l2_coef = float(action_l2_coef)
        self.cql_alpha_reward = float(cql_alpha_reward)
        self.cql_alpha_cost = float(cql_alpha_cost)
        self.warmup_actor_steps = int(warmup_actor_steps)
        self.total_updates = 0

        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.actor_model_name = actor_model_name
        self.critic_model_name = critic_model_name
        self.actor_hidden_dims = tuple(actor_hidden_dims)
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.critic_attend_heads = int(critic_attend_heads)

        self.behavior = build_behavior_model(
            obs_dim, act_dim, latent_dim=latent_dim, hidden_dims=self.actor_hidden_dims
        ).to(self.device)
        self.actor = build_actor_model(
            actor_model_name, obs_dim, act_dim, action_low, action_high, phi=phi,
            hidden_dims=self.actor_hidden_dims
        ).to(self.device)
        self.qr1 = build_critic_model(
            critic_model_name, state_dim, n_agents, act_dim,
            hidden_dims=self.critic_hidden_dims, attend_heads=critic_attend_heads
        ).to(self.device)
        self.qr2 = build_critic_model(
            critic_model_name, state_dim, n_agents, act_dim,
            hidden_dims=self.critic_hidden_dims, attend_heads=critic_attend_heads
        ).to(self.device)
        self.qc1 = build_critic_model(
            critic_model_name, state_dim, n_agents, act_dim,
            hidden_dims=self.critic_hidden_dims, attend_heads=critic_attend_heads
        ).to(self.device)
        self.qc2 = build_critic_model(
            critic_model_name, state_dim, n_agents, act_dim,
            hidden_dims=self.critic_hidden_dims, attend_heads=critic_attend_heads
        ).to(self.device)

        self.behavior_targ = copy.deepcopy(self.behavior).eval()
        self.actor_targ = copy.deepcopy(self.actor).eval()
        self.qr1_targ = copy.deepcopy(self.qr1).eval()
        self.qr2_targ = copy.deepcopy(self.qr2).eval()
        self.qc1_targ = copy.deepcopy(self.qc1).eval()
        self.qc2_targ = copy.deepcopy(self.qc2).eval()

        self.behavior_opt = torch.optim.Adam(self.behavior.parameters(), lr=vae_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.qr_opt = torch.optim.Adam(list(self.qr1.parameters()) + list(self.qr2.parameters()), lr=critic_lr)
        self.qc_opt = torch.optim.Adam(list(self.qc1.parameters()) + list(self.qc2.parameters()), lr=critic_lr)

        self.log_lam = nn.Parameter(torch.tensor(-2.0, device=self.device))
        self.lam_opt = torch.optim.Adam([self.log_lam], lr=lag_lr)

    @property
    def lam(self):
        return F.softplus(self.log_lam)

    def _reshape_agent_batch(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n, dim = x.shape
        return x.reshape(bsz * n, dim)

    def _current_bc_coef(self) -> float:
        if self.total_updates <= self.warmup_actor_steps:
            return self.bc_coef
        span = max(1, self.warmup_actor_steps)
        progress = min(1.0, (self.total_updates - self.warmup_actor_steps) / float(10 * span))
        coef = self.bc_coef_end + 0.5 * (self.bc_coef - self.bc_coef_end) * (
            1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item()
        )
        return float(coef)

    def _joint_action_from_local(
        self,
        obs: torch.Tensor,
        state: torch.Tensor = None,
        use_target: bool = False,
        deterministic: bool = False,
        num_candidates: int = 1,
    ) -> torch.Tensor:
        bsz, n, _ = obs.shape
        flat_obs = self._reshape_agent_batch(obs)
        behavior_net = self.behavior_targ if use_target else self.behavior
        actor_net = self.actor_targ if use_target else self.actor

        if not use_target:
            num_candidates = 1

        if num_candidates == 1 or state is None:
            if use_target:
                with torch.no_grad():
                    bc = behavior_net.decode(flat_obs, deterministic=deterministic)
                    act = actor_net(flat_obs, bc)
            else:
                bc = behavior_net.decode(flat_obs, deterministic=deterministic)
                act = actor_net(flat_obs, bc)
            return act.reshape(bsz, n, self.act_dim)

        cand_actions = []
        cand_scores = []
        for _ in range(num_candidates):
            with torch.no_grad():
                bc = behavior_net.decode(flat_obs, deterministic=False)
                act = actor_net(flat_obs, bc).reshape(bsz, n, self.act_dim)
                qr = (self.qr1_targ if use_target else self.qr1)(state, act)
                qc = (self.qc1_targ if use_target else self.qc1)(state, act)
                score = qr - self.lam.detach() * qc
            cand_actions.append(act)
            cand_scores.append(score)

        q_stack = torch.stack(cand_scores, dim=0).squeeze(-1)
        best_idx = torch.argmax(q_stack, dim=0)
        out = []
        for b in range(bsz):
            out.append(cand_actions[int(best_idx[b].item())][b])
        return torch.stack(out, dim=0)

    def _sample_random_joint_actions(self, batch_size: int) -> torch.Tensor:
        u = torch.rand(batch_size, self.n_agents, self.act_dim, device=self.device)
        return self.action_low + (self.action_high - self.action_low) * u

    def _cql_penalty(
        self,
        critic: nn.Module,
        state: torch.Tensor,
        obs: torch.Tensor,
        data_actions: torch.Tensor,
        num_samples: int = 5
    ) -> torch.Tensor:
        bsz = obs.shape[0]
        random_qs = []
        policy_qs = []
        for _ in range(num_samples):
            rand_a = self._sample_random_joint_actions(bsz)
            random_qs.append(critic(state, rand_a))
            with torch.no_grad():
                pi_a = self._joint_action_from_local(
                    obs, state=state, use_target=False, deterministic=False, num_candidates=1
                )
            policy_qs.append(critic(state, pi_a))

        random_q = torch.cat(random_qs, dim=1)
        policy_q = torch.cat(policy_qs, dim=1)
        data_q = critic(state, data_actions)
        all_q = torch.cat([random_q, policy_q], dim=1)
        return torch.logsumexp(all_q, dim=1, keepdim=True).mean() - data_q.mean()

    @torch.no_grad()
    def select_action(self, obs, state=None, deterministic: bool = False, num_candidates: int = 20):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_t = None if state is None else torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        act = self._joint_action_from_local(
            obs_t, state_t, use_target=False, deterministic=deterministic, num_candidates=num_candidates
        )[0]
        return act.cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.total_updates += 1
        obs = batch["obs"].to(self.device)
        state = batch["state"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        costs = batch["costs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        dones = batch["dones"].to(self.device)

        bsz, n, _ = obs.shape
        flat_obs = self._reshape_agent_batch(obs)
        flat_act = self._reshape_agent_batch(actions)

        recon, mu, std = self.behavior(flat_obs, flat_act)
        recon_loss = F.mse_loss(recon, flat_act)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2) + 1e-8) - mu.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss
        self.behavior_opt.zero_grad()
        vae_loss.backward()
        self.behavior_opt.step()

        with torch.no_grad():
            next_joint_action = self._joint_action_from_local(
                next_obs, state=next_state, use_target=True, deterministic=False, num_candidates=10
            )
            qr_targ = torch.min(
                self.qr1_targ(next_state, next_joint_action),
                self.qr2_targ(next_state, next_joint_action)
            )
            qc_targ = torch.max(
                self.qc1_targ(next_state, next_joint_action),
                self.qc2_targ(next_state, next_joint_action)
            )
            target_reward_q = rewards + self.gamma * (1.0 - dones) * qr_targ
            target_cost_q = costs + self.gamma * (1.0 - dones) * qc_targ

        qr1_pred = self.qr1(state, actions)
        qr2_pred = self.qr2(state, actions)
        qr_td_loss = F.mse_loss(qr1_pred, target_reward_q) + F.mse_loss(qr2_pred, target_reward_q)
        qr_cql_loss = self._cql_penalty(self.qr1, state, obs, actions) + self._cql_penalty(self.qr2, state, obs, actions)
        qr_loss = qr_td_loss + self.cql_alpha_reward * qr_cql_loss
        self.qr_opt.zero_grad()
        qr_loss.backward()
        self.qr_opt.step()

        qc1_pred = self.qc1(state, actions)
        qc2_pred = self.qc2(state, actions)
        qc_td_loss = F.mse_loss(qc1_pred, target_cost_q) + F.mse_loss(qc2_pred, target_cost_q)
        qc_cql_loss = self._cql_penalty(self.qc1, state, obs, actions) + self._cql_penalty(self.qc2, state, obs, actions)
        qc_loss = qc_td_loss + self.cql_alpha_cost * qc_cql_loss
        self.qc_opt.zero_grad()
        qc_loss.backward()
        self.qc_opt.step()

        joint_pi = self._joint_action_from_local(
            obs,
            state=None,
            use_target=False,
            deterministic=False,
            num_candidates=1,
        )
        qr_pi = self.qr1(state, joint_pi)
        qc_pi = self.qc1(state, joint_pi)

        with torch.no_grad():
            flat_bc = self.behavior.decode(flat_obs, deterministic=True)

        flat_pi = joint_pi.reshape(bsz * n, -1)
        bc_reg = F.mse_loss(flat_pi, flat_bc)
        act_l2 = (flat_pi ** 2).mean()
        bc_coef_now = self._current_bc_coef()

        if self.total_updates <= self.warmup_actor_steps:
            actor_loss = bc_coef_now * bc_reg + self.action_l2_coef * act_l2
        else:
            actor_obj = qr_pi - self.lam.detach() * qc_pi
            actor_loss = -actor_obj.mean() + bc_coef_now * bc_reg + self.action_l2_coef * act_l2

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        lam_loss = -(self.lam * (qc_pi.detach() - self.cost_limit)).mean()
        self.lam_opt.zero_grad()
        lam_loss.backward()
        self.lam_opt.step()

        self._soft_update(self.behavior, self.behavior_targ, self.tau)
        self._soft_update(self.actor, self.actor_targ, self.tau)
        self._soft_update(self.qr1, self.qr1_targ, self.tau)
        self._soft_update(self.qr2, self.qr2_targ, self.tau)
        self._soft_update(self.qc1, self.qc1_targ, self.tau)
        self._soft_update(self.qc2, self.qc2_targ, self.tau)

        return {
            "vae_loss": float(vae_loss.item()),
            "qr_loss": float(qr_loss.item()),
            "qc_loss": float(qc_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "lambda": float(self.lam.item()),
            "qr_pi": float(qr_pi.mean().item()),
            "qc_pi": float(qc_pi.mean().item()),
            "bc_coef_now": float(bc_coef_now),
            "qr_cql": float(qr_cql_loss.item()),
            "qc_cql": float(qc_cql_loss.item()),
        }

    @staticmethod
    def _soft_update(src, tgt, tau: float = 0.005):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def save(self, path: str) -> None:
        ckpt = {
            "behavior": self.behavior.state_dict(),
            "actor": self.actor.state_dict(),
            "qr1": self.qr1.state_dict(),
            "qr2": self.qr2.state_dict(),
            "qc1": self.qc1.state_dict(),
            "qc2": self.qc2.state_dict(),
            "log_lam": self.log_lam.data,
            "meta": {
                "n_agents": self.n_agents,
                "act_dim": self.act_dim,
                "cost_limit": self.cost_limit,
                "actor_model_name": self.actor_model_name,
                "critic_model_name": self.critic_model_name,
                "actor_hidden_dims": list(self.actor_hidden_dims),
                "critic_hidden_dims": list(self.critic_hidden_dims),
                "critic_attend_heads": self.critic_attend_heads,
            },
        }
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.behavior.load_state_dict(ckpt["behavior"])
        self.actor.load_state_dict(ckpt["actor"])
        self.qr1.load_state_dict(ckpt["qr1"])
        self.qr2.load_state_dict(ckpt["qr2"])
        self.qc1.load_state_dict(ckpt["qc1"])
        self.qc2.load_state_dict(ckpt["qc2"])
        self.log_lam.data.copy_(ckpt["log_lam"])