# offline_safe/dataset/policies.py
import numpy as np
import torch

from models.model_registry import Model
from utilities.util import prep_obs, translate_action


class RandomPolicy:
    def __init__(self, env):
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.n_agents = env.get_num_of_agents()
        self.action_dim = env.get_total_actions()

    def reset(self):
        pass

    def act(self, obs, state, env):
        # 直接输出环境动作范围内的“actual action”
        a = np.random.uniform(
            low=self.low,
            high=self.high,
            size=(self.n_agents, self.action_dim),
        ).astype(np.float32)
        return a


class NoisyDroopPolicy:
    """
    先用一个简单 droop-like 规则构造 medium，
    后续你可替换成 traditional_control 中的 Matlab 轨迹。
    """
    def __init__(self, env, gain=6.0, noise_std=0.03):
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.n_agents = env.get_num_of_agents()
        self.action_dim = env.get_total_actions()
        self.gain = gain
        self.noise_std = noise_std

    def reset(self):
        pass

    def act(self, obs, state, env):
        bus_v = env._get_voltage()  # [n_bus]
        sgen_bus = env.powergrid.sgen["bus"].to_numpy(copy=True)
        local_v = bus_v[sgen_bus]   # distributed: 每个 agent 对应一个 sgen

        # 电压高 -> 吸收无功更多；电压低 -> 注入更多
        action = self.gain * (1.0 - local_v)
        action += np.random.normal(0.0, self.noise_std, size=action.shape)
        action = np.clip(action, self.low, self.high)

        return action.reshape(self.n_agents, self.action_dim).astype(np.float32)


class ExpertMATD3Policy:
    """
    直接对齐 test.py / PGTester.run() 的推理逻辑
    """
    def __init__(self, args, checkpoint_path, alg="matd3"):
        self.args = args
        self.device = torch.device("cpu")

        model_cls = Model[alg]
        if args.target:
            target_net = model_cls(args)
            self.behaviour_net = model_cls(args, target_net)
        else:
            self.behaviour_net = model_cls(args)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.behaviour_net.load_state_dict(checkpoint["model_state_dict"])
        self.behaviour_net.eval()

        self.last_hid = None

    def reset(self):
        # 与 PGTester.run() 一致
        self.last_hid = self.behaviour_net.policy_dicts[0].init_hidden()

    @torch.no_grad()
    def act(self, obs, state, env):
        """
        返回 actual action，范围已经被 translate_action 映射到环境动作区间
        shape: [n_agents, action_dim]
        """
        state_ = prep_obs(obs).contiguous().view(
            1, self.args.agent_num, self.args.obs_size
        ).to(self.device)

        action, _, _, _, hid = self.behaviour_net.get_actions(
            state_,
            status="test",
            exploration=False,
            actions_avail=torch.tensor(env.get_avail_actions()),
            target=False,
            last_hid=self.last_hid,
        )

        _, actual = translate_action(self.args, action, env)
        self.last_hid = hid

        actual = np.asarray(actual, dtype=np.float32).reshape(
            self.args.agent_num, self.args.action_dim
        )
        return actual