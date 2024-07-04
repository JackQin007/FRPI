from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from FRPI.agent.frpi import FRPIAgent, FRPIParams
from FRPI.algorithm.base import Algorithm
from FRPI.utils.experience import Experience


class FRPIAlgState(NamedTuple):
    qf_opt_state: optax.OptState
    policy_opt_state: optax.OptState


class FRPI(Algorithm):
    def __init__(
        self,
        agent: FRPIAgent,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.optim = optax.adam(lr)
        self.alg_state = FRPIAlgState(
            qf_opt_state=self.optim.init(agent.params.qf),
            policy_opt_state=self.optim.init(agent.params.policy),
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray,
            params: FRPIParams,
            alg_state: FRPIAlgState,
            data: Experience
        ) -> Tuple[FRPIParams, FRPIAlgState, dict]:
            obs, action, next_obs, next_cost, done = (
                data.obs,
                data.action,
                data.next_obs,
                data.next_cost,
                data.done,
            )
            qf_params, target_qf_params, policy_params = params
            qf_opt_state, policy_opt_state = alg_state

            # update qf
            next_action = self.agent.policy(policy_params, next_obs)
            next_qf_target = jax.nn.sigmoid(self.agent.q(
                target_qf_params, next_obs, next_action))
            done = done.astype(jnp.float32)
            next_cost = (next_cost > 0).astype(jnp.float32)
            next_cost = -next_cost
            qf_backup = done + next_cost + (1 - done) * (1 + next_cost) * \
                self.gamma * next_qf_target

            def qf_loss_fn(qf_params: hk.Params):
                qf = self.agent.q(qf_params, obs, action)
                qf_loss = optax.sigmoid_binary_cross_entropy(qf, qf_backup).mean()
                return qf_loss, qf

            (qf_loss, qf), qf_grads = jax.value_and_grad(
                qf_loss_fn, has_aux=True)(qf_params)
            qf_updates, qf_opt_state = self.optim.update(qf_grads, qf_opt_state)
            qf_params = optax.apply_updates(qf_params, qf_updates)

            # update policy
            def policy_loss_fn(policy_params: hk.Params) -> jnp.ndarray:
                new_action = self.agent.policy(policy_params, obs)
                qf = self.agent.q(qf_params, obs, new_action)
                policy_loss = qf.mean()
                return policy_loss

            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
            policy_updates, policy_opt_state = self.optim.update(
                policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_updates)

            # update target networks
            target_qf_params = optax.incremental_update(
                qf_params, target_qf_params, self.tau)

            params = FRPIParams(
                qf=qf_params,
                target_qf=target_qf_params,
                policy=policy_params,
            )
            alg_state = FRPIAlgState(
                qf_opt_state=qf_opt_state,
                policy_opt_state=policy_opt_state,
            )
            info = {
                'qf_loss': qf_loss,
                'qf': jax.nn.sigmoid(qf).mean(),
                'policy_loss': policy_loss,
            }
            return params, alg_state, info

        self.stateless_update = stateless_update
