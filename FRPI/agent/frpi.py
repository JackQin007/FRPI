from typing import NamedTuple, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Normal

from FRPI.agent.base import Agent
from FRPI.agent.block import QNet, DeterministicPolicyNet


class FRPIParams(NamedTuple):
    qf: hk.Params
    target_qf: hk.Params
    policy: hk.Params


class FRPIAgent(Agent):
    def __init__(
        self,
        key: jax.random.KeyArray,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        act_noise: float = 0.1,
    ):
        def q_fn(obs, act):
            return QNet(hidden_sizes)(obs, act)

        def policy_fn(obs):
            return DeterministicPolicyNet(
                act_dim, hidden_sizes, output_activation=jax.nn.tanh)(obs)

        q = hk.without_apply_rng(hk.transform(q_fn))
        policy = hk.without_apply_rng(hk.transform(policy_fn))

        qf_key, policy_key = jax.random.split(key, 2)
        obs = jnp.zeros((1, obs_dim))
        act = jnp.zeros((1, act_dim))
        qf_params = q.init(qf_key, obs, act)
        target_qf_params = qf_params
        policy_params = policy.init(policy_key, obs)
        self.params = FRPIParams(
            qf=qf_params,
            target_qf=target_qf_params,
            policy=policy_params,
        )

        self.q = q.apply
        self.policy = policy.apply
        self.act_noise = act_noise

    def get_action(self, key: jax.random.KeyArray, obs: np.ndarray) -> np.ndarray:
        act = self.policy(self.params.policy, obs)
        noise = Normal(jnp.zeros_like(act), self.act_noise).sample(key)
        act = jnp.clip(act + noise, -1, 1)
        return np.asarray(act)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.policy(self.params.policy, obs))

    def get_feasibility(self, obs: np.ndarray) -> np.ndarray:
        act = self.policy(self.params.policy, obs)
        qf = jax.nn.sigmoid(self.q(self.params.qf, obs, act))
        return np.asarray(qf)
