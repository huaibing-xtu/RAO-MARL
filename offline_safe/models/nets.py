import math
import torch
import torch.nn as nn


def mlp(in_dim, hidden_dims, out_dim, act=nn.ReLU, out_act=None):
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(act())
        last = h
    layers.append(nn.Linear(last, out_dim))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.norm2(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class GatedResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        value, gate = torch.chunk(self.fc(h), 2, dim=-1)
        h = self.act(value) * torch.sigmoid(gate)
        h = self.proj(self.drop(h))
        return x + h


class SharedBehaviorVAE(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim=16, hidden_dims=(256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.encoder = mlp(obs_dim + act_dim, hidden_dims, latent_dim * 2)
        self.decoder = mlp(obs_dim + latent_dim, hidden_dims, act_dim)

    def encode(self, obs, act):
        h = self.encoder(torch.cat([obs, act], dim=-1))
        mu, log_std = torch.chunk(h, 2, dim=-1)
        log_std = torch.clamp(log_std, -4.0, 15.0)
        std = torch.exp(log_std)
        return mu, std

    def decode(self, obs, z=None, deterministic=False):
        if z is None:
            if deterministic:
                z = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
            else:
                z = torch.randn(obs.shape[0], self.latent_dim, device=obs.device)
        return self.decoder(torch.cat([obs, z], dim=-1))

    def forward(self, obs, act):
        mu, std = self.encode(obs, act)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(obs, z)
        return recon, mu, std


class SharedPerturbActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, phi=0.05, hidden_dims=(256, 256)):
        super().__init__()
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.phi = phi
        self.net = mlp(obs_dim + act_dim, hidden_dims, act_dim, out_act=nn.Tanh)

    def forward(self, obs, bc_action):
        delta = self.phi * (self.action_high - self.action_low) * self.net(torch.cat([obs, bc_action], dim=-1))
        action = bc_action + delta
        return torch.clamp(action, self.action_low, self.action_high)


class ResidualPerturbActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, phi=0.05, hidden_dims=(256, 256)):
        super().__init__()
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.phi = phi
        width = hidden_dims[0] if len(hidden_dims) > 0 else 256
        depth = max(2, len(hidden_dims))
        self.in_proj = nn.Linear(obs_dim + act_dim, width)
        self.blocks = nn.ModuleList([ResidualMLPBlock(width, dropout=0.05) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, act_dim)

    def forward(self, obs, bc_action):
        x = torch.cat([obs, bc_action], dim=-1)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        delta = torch.tanh(self.out(h))
        delta = self.phi * (self.action_high - self.action_low) * delta
        action = bc_action + delta
        return torch.clamp(action, self.action_low, self.action_high)


class GatedResidualPerturbActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, phi=0.05, hidden_dims=(256, 256)):
        super().__init__()
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.phi = phi
        width = hidden_dims[0] if len(hidden_dims) > 0 else 256
        depth = max(2, len(hidden_dims))
        self.in_proj = nn.Linear(obs_dim + act_dim, width)
        self.blocks = nn.ModuleList([GatedResidualMLPBlock(width, dropout=0.05) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs, bc_action):
        x = torch.cat([obs, bc_action], dim=-1)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        delta = self.phi * (self.action_high - self.action_low) * self.head(h)
        action = bc_action + delta
        return torch.clamp(action, self.action_low, self.action_high)


class CentralCritic(nn.Module):
    def __init__(self, state_dim, n_agents, act_dim, hidden_dims=(512, 512)):
        super().__init__()
        self.net = mlp(state_dim + n_agents * act_dim, hidden_dims, 1)

    def forward(self, state, joint_action):
        x = torch.cat([state, joint_action.reshape(joint_action.shape[0], -1)], dim=-1)
        return self.net(x)


class AttentionCentralCritic(nn.Module):
    def __init__(self, state_dim, n_agents, act_dim, hidden_dims=(256, 256), attend_heads=4):
        super().__init__()
        hidden_dim = hidden_dims[0] if len(hidden_dims) > 0 else 256
        if hidden_dim % attend_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by attend_heads ({attend_heads})")
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.act_dim = act_dim
        state_chunk = max(1, (state_dim + n_agents - 1) // n_agents)
        self.state_embed = nn.Linear(state_chunk + act_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, attend_heads, batch_first=True)
        self.post = mlp(hidden_dim * 2, hidden_dims[1:] if len(hidden_dims) > 1 else (hidden_dim,), 1)

    def _split_state(self, state):
        chunks = torch.chunk(state, self.n_agents, dim=-1)
        if len(chunks) < self.n_agents:
            pad = [chunks[-1]] * (self.n_agents - len(chunks))
            chunks = chunks + tuple(pad)
        max_dim = max(chunk.shape[-1] for chunk in chunks)
        out = []
        for chunk in chunks[:self.n_agents]:
            if chunk.shape[-1] < max_dim:
                pad = torch.zeros(chunk.shape[0], max_dim - chunk.shape[-1], device=chunk.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=-1)
            out.append(chunk)
        return torch.stack(out, dim=1)

    def forward(self, state, joint_action):
        state_per_agent = self._split_state(state)
        sa = torch.cat([state_per_agent, joint_action], dim=-1)
        token = torch.relu(self.state_embed(sa))
        attended, _ = self.attn(token, token, token, need_weights=False)
        pooled = torch.cat([token.mean(dim=1), attended.mean(dim=1)], dim=-1)
        return self.post(pooled)


class TransformerCriticBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerCentralCritic(nn.Module):
    def __init__(self, state_dim, n_agents, act_dim, hidden_dims=(512, 512), attend_heads=4):
        super().__init__()
        hidden_dim = hidden_dims[0] if len(hidden_dims) > 0 else 512
        if hidden_dim % attend_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by attend_heads ({attend_heads})")
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.state_chunk = math.ceil(state_dim / n_agents)

        self.agent_embed = nn.Embedding(n_agents, hidden_dim)
        self.state_proj = nn.Linear(self.state_chunk, hidden_dim)
        self.action_proj = nn.Linear(act_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        depth = max(2, len(hidden_dims))
        self.blocks = nn.ModuleList(
            [TransformerCriticBlock(hidden_dim, attend_heads, dropout=0.1) for _ in range(depth)]
        )

        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _split_state(self, state):
        total_needed = self.state_chunk * self.n_agents
        if state.shape[-1] < total_needed:
            pad = torch.zeros(state.shape[0], total_needed - state.shape[-1], device=state.device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=-1)
        elif state.shape[-1] > total_needed:
            state = state[:, :total_needed]
        return state.view(state.shape[0], self.n_agents, self.state_chunk)

    def forward(self, state, joint_action):
        state_tokens = self._split_state(state)
        s = self.state_proj(state_tokens)
        a = self.action_proj(joint_action)
        x = self.fuse(torch.cat([s, a], dim=-1))

        ids = torch.arange(self.n_agents, device=state.device)
        x = x + self.agent_embed(ids).unsqueeze(0)

        for blk in self.blocks:
            x = blk(x)

        pooled_mean = x.mean(dim=1)
        pooled_max = x.max(dim=1).values
        return self.readout(torch.cat([pooled_mean, pooled_max], dim=-1))