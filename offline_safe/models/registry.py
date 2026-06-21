from typing import Sequence

from offline_safe.models.nets import (
    SharedBehaviorVAE,
    SharedPerturbActor,
    ResidualPerturbActor,
    GatedResidualPerturbActor,
    CentralCritic,
    AttentionCentralCritic,
    TransformerCentralCritic,
)


ACTOR_REGISTRY = {
    "mlp": SharedPerturbActor,
    "residual": ResidualPerturbActor,
    "gated_residual": GatedResidualPerturbActor,
}

CRITIC_REGISTRY = {
    "mlp": CentralCritic,
    "central": CentralCritic,
    "maac": AttentionCentralCritic,
    "maac_plus": TransformerCentralCritic,
    "transformer": TransformerCentralCritic,
}


def build_behavior_model(
    obs_dim: int,
    act_dim: int,
    latent_dim: int = 16,
    hidden_dims: Sequence[int] = (256, 256),
):
    return SharedBehaviorVAE(obs_dim, act_dim, latent_dim=latent_dim, hidden_dims=tuple(hidden_dims))


def build_actor_model(
    name: str,
    obs_dim: int,
    act_dim: int,
    action_low: float,
    action_high: float,
    phi: float = 0.05,
    hidden_dims: Sequence[int] = (256, 256),
):
    name = name.lower()
    if name not in ACTOR_REGISTRY:
        raise ValueError(f"Unsupported actor model: {name}. Available: {sorted(ACTOR_REGISTRY)}")
    actor_cls = ACTOR_REGISTRY[name]
    return actor_cls(
        obs_dim,
        act_dim,
        action_low,
        action_high,
        phi=phi,
        hidden_dims=tuple(hidden_dims),
    )


def build_critic_model(
    name: str,
    state_dim: int,
    n_agents: int,
    act_dim: int,
    hidden_dims: Sequence[int] = (512, 512),
    attend_heads: int = 4,
):
    name = name.lower()
    if name not in CRITIC_REGISTRY:
        raise ValueError(f"Unsupported critic model: {name}. Available: {sorted(CRITIC_REGISTRY)}")
    critic_cls = CRITIC_REGISTRY[name]
    if name in {"maac", "maac_plus", "transformer"}:
        return critic_cls(
            state_dim,
            n_agents,
            act_dim,
            hidden_dims=tuple(hidden_dims),
            attend_heads=attend_heads,
        )
    return critic_cls(state_dim, n_agents, act_dim, hidden_dims=tuple(hidden_dims))