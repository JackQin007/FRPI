import math
from typing import NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from numpyro.distributions import Normal
import numpy as np

from FRPI.agent.base import Agent
from FRPI.agent.block import QNet, StochasticPolicyNet,DeterministicPolicyNet


class FRPISACParams(NamedTuple):
    # FRI params
    qf: hk.Params
    target_qf: hk.Params

    # SAC params
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params


class FRPISACAgent(Agent):
    def __init__(
        self,
        key: jax.random.KeyArray,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
    ):
        def q_fn(obs, act):
            return QNet(hidden_sizes)(obs, act)

        def policy_fn(obs):
            return StochasticPolicyNet(act_dim, hidden_sizes)(obs)
            # return DeterministicPolicyNet(act_dim, hidden_sizes)(obs)

        qf = hk.without_apply_rng(hk.transform(q_fn))
        q = hk.without_apply_rng(hk.transform(q_fn))
        policy = hk.without_apply_rng(hk.transform(policy_fn))

        qf_key, q1_key, q2_key, policy_key = jax.random.split(key, 4)
        obs = jnp.zeros((1, obs_dim))
        act = jnp.zeros((1, act_dim))
        qf_params = qf.init(qf_key, obs, act)
        target_qf_params = qf_params
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)
        self.params = FRPISACParams(
            qf=qf_params,
            target_qf=target_qf_params,
            q1=q1_params,
            q2=q2_params,
            target_q1=target_q1_params,
            target_q2=target_q2_params,
            policy=policy_params,
        )

        self.qf = qf.apply
        self.q = q.apply
        self.policy = policy.apply
        self.act_dim = act_dim

    def get_action(self, key: jax.random.KeyArray, obs: np.ndarray) -> np.ndarray:
        mean, std = self.policy(self.params.policy, obs)
        z = Normal(mean, std).sample(key)
        return np.asarray(jnp.tanh(z))

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        try:
            mean, _ = self.policy(self.params.policy, obs)
        except Exception as e:
            mean = self.policy(self.params.policy, obs)
        return np.asarray(jnp.tanh(mean))

    def evaluate(
        self, key: jax.random.KeyArray, policy_params: hk.Params, obs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        try:
            mean, std = self.policy(policy_params, obs)
            dist = Normal(mean, std)
            z = dist.rsample(key)
            act = jnp.tanh(z)
            logp = (dist.log_prob(z) - 2 * (math.log(2) -
                    z - jax.nn.softplus(-2 * z))).sum(axis=-1)
            return act, logp
        except Exception as e:
            z = self.policy(policy_params, obs)
            act = jnp.tanh(z)
            return act

    def get_feasibility(self, obs: np.ndarray) -> np.ndarray:
        mean, _ = self.policy(self.params.policy, obs)
        act = jnp.tanh(mean)
        qf = jax.nn.tanh(self.qf(self.params.qf, obs, act))
        return np.asarray(qf)

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        mean, _ = self.policy(self.params.policy, obs)
        act = jnp.tanh(mean)
        q = self.q(self.params.q1, obs, act)
        return np.asarray(q)
