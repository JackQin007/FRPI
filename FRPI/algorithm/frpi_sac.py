import math
from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from FRPI.agent.frpi_sac import FRPISACAgent, FRPISACParams
from cpo.algorithm.sac import SAC
from FRPI.utils.experience import Experience
from FRPI.utils.math import masked_mean


class FRPISACAlgState(NamedTuple):
    qf_opt_state: optax.OptState
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    policy_opt_state: optax.OptState
    log_alpha: jnp.ndarray
    log_alpha_opt_state: optax.OptState
    t: float
    step: int


class FRPISAC(SAC):
    def __init__(
        self,
        agent: FRPISACAgent,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        alpha: float = 1.0,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        pf: float = 0.1,
        eps: float = 1e-6,
        init_t: float = 1.0,
        t_increase_factor: float = 1.1,
        t_update_delay: int = 1000,
        max_t: Optional[float] = None,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        log_alpha = jnp.array(math.log(alpha), dtype=jnp.float32)
        if target_entropy is None:
            self.target_entropy = -self.agent.act_dim
        else:
            self.target_entropy = target_entropy
        self.pf_logit = -math.log(1 / pf - 1)
        self.eps = eps
        self.t_increase_factor = t_increase_factor
        self.t_update_delay = t_update_delay
        if max_t is None:
            self.max_t = math.inf
        else:
            self.max_t = max_t
        self.optim = optax.adam(lr)
        self.alg_state = FRPISACAlgState(
            qf_opt_state=self.optim.init(agent.params.qf),
            q1_opt_state=self.optim.init(agent.params.q1),
            q2_opt_state=self.optim.init(agent.params.q2),
            policy_opt_state=self.optim.init(agent.params.policy),
            log_alpha=log_alpha,
            log_alpha_opt_state=self.optim.init(log_alpha),
            t=init_t,
            step=0,
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray,
            params: FRPISACParams,
            alg_state: FRPISACAlgState,
            data: Experience
        ) -> Tuple[FRPISACParams, FRPISACAlgState, dict]:
            obs, action, reward, next_obs, next_cost, done, next_goal= (
                data.obs,
                data.action,
                data.reward,
                data.next_obs,
                data.next_cost,
                data.done,
                data.next_goal,
            )
            (
                qf_params,
                target_qf_params,
                q1_params,
                q2_params,
                target_q1_params,
                target_q2_params,
                policy_params,
            ) = params
            (
                qf_opt_state,
                q1_opt_state,
                q2_opt_state,
                policy_opt_state,
                log_alpha,
                log_alpha_opt_state,
                t,
                step,
            ) = alg_state
            key_qf, key_q, key_policy = jax.random.split(key, 3)

            # update qf
            try:
                next_action, _ = self.agent.evaluate(key_qf, policy_params, next_obs)
            except Exception as e:
                next_action = self.agent.evaluate(key_qf, policy_params, next_obs)
            next_qf_target = jax.nn.tanh(self.agent.qf(
                target_qf_params, next_obs, next_action))
            done = done.astype(jnp.float32)
            next_goal = next_goal.astype(jnp.float32)
            next_cost = (next_cost > 0).astype(jnp.float32)
            next_cost = -next_cost
            qf_backup = next_cost + next_goal + (1 - done) * (1 + next_cost) *(1 - next_goal)* \
                self.gamma * next_qf_target

            def qf_loss_fn(qf_params: hk.Params):
                qf = self.agent.qf(qf_params, obs, action)
                # qf_loss = optax.sigmoid_binary_cross_entropy(qf, qf_backup).mean()
                qf_loss = ((qf - qf_backup) ** 2).mean()
                return qf_loss, qf

            (qf_loss, qf), qf_grads = jax.value_and_grad(
                qf_loss_fn, has_aux=True)(qf_params)
            qf_updates, qf_opt_state = self.optim.update(qf_grads, qf_opt_state)
            qf_params = optax.apply_updates(qf_params, qf_updates)

            # update q
            try:
                next_action, next_logp = self.agent.evaluate(key_q, policy_params, next_obs)
            except Exception as e:
                next_action = self.agent.evaluate(key_q, policy_params, next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            try:
                q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            except Exception as e:
                q_target = jnp.minimum(q1_target, q2_target) 
            q_backup = reward + (1 - done) * self.gamma * q_target
            qf = self.agent.qf(qf_params, obs, action)
            qf_tanh = jax.nn.tanh(qf)
            qf_mask = qf_tanh > self.eps

            def q_loss_fn(q_params: hk.Params):
                q = self.agent.q(q_params, obs, action)
                q_loss = masked_mean((q - q_backup) ** 2, qf_mask)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(
                q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(
                q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # update policy
            def policy_loss_fn(policy_params: hk.Params):
                try:
                    new_action, new_logp = self.agent.evaluate(key_policy, policy_params, obs)
                except Exception as e:
                    new_action = self.agent.evaluate(key_policy, policy_params, obs)
                q1 = self.agent.q(q1_params, obs, new_action)
                q2 = self.agent.q(q2_params, obs, new_action)
                q = jnp.minimum(q1, q2)

                qf = self.agent.qf(qf_params, obs, new_action)
                log_barrier = -jnp.log(jnp.maximum(qf_tanh+2*self.eps , self.eps))
                # feasible = qf - self.pf_logit < -self.eps
                feasible = qf_tanh  >  self.eps
                try:
                    policy_loss1 = feasible * (jnp.exp(log_alpha) * new_logp - q +
                                            1 / t * log_barrier)
                    policy_loss2 = ~feasible * -qf
                    policy_loss = (policy_loss1 + policy_loss2).mean()

                    new_logp = masked_mean(new_logp, feasible)

                    return policy_loss, (policy_loss1, policy_loss2, log_barrier, 
                                        feasible, new_logp)
                except Exception as e:
                    policy_loss1 = feasible * ( - q +
                                            1 / t * log_barrier)
                    policy_loss2 = ~feasible * -qf
                    policy_loss = (policy_loss1 + policy_loss2).mean()
                    return policy_loss, (policy_loss1, policy_loss2, log_barrier, 
                                        feasible)
                

            (policy_loss, aux), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True)(policy_params)
            try:
                (policy_loss1, policy_loss2, log_barrier, feasible, new_logp) = aux
            except Exception as e:
                (policy_loss1, policy_loss2, log_barrier, feasible) = aux
                
            policy_update, policy_opt_state = self.optim.update(
                policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # update alpha
            try:
                log_alpha, log_alpha_opt_state = self.update_alpha(
                    log_alpha, log_alpha_opt_state, new_logp)
            except Exception as e:
                pass
            # update target networks
            target_qf_params = optax.incremental_update(
                qf_params, target_qf_params, self.tau)
            target_q1_params = optax.incremental_update(
                q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(
                q2_params, target_q2_params, self.tau)

            # update logarithmic barrier coefficient
            t = jax.lax.cond(
                (step + 1) % self.t_update_delay == 0,
                lambda x: jnp.minimum(self.max_t, self.t_increase_factor * x),
                lambda x: x,
                t,
            )

            params = FRPISACParams(
                qf=qf_params,
                target_qf=target_qf_params,
                q1=q1_params,
                q2=q2_params,
                target_q1=target_q1_params,
                target_q2=target_q2_params,
                policy=policy_params,
            )
            alg_state = FRPISACAlgState(
                qf_opt_state=qf_opt_state,
                q1_opt_state=q1_opt_state,
                q2_opt_state=q2_opt_state,
                policy_opt_state=policy_opt_state,
                log_alpha=log_alpha,
                log_alpha_opt_state=log_alpha_opt_state,
                t=t,
                step=step + 1,
            )
            try:
                info = {
                    'qf_loss': qf_loss,
                    'qf': jax.nn.sigmoid(qf).mean(),
                    'q1_loss': q1_loss,
                    'q2_loss': q2_loss,
                    'q1': q1.mean(),
                    'q2': q2.mean(),
                    'policy_loss': policy_loss,
                    'feasible_policy_loss': policy_loss1.mean(),
                    'infeasible_policy_loss': policy_loss2.mean(),
                    'log_barrier': log_barrier.mean(),
                    'feasible_ratio': feasible.mean(),
                    'entropy': -masked_mean(new_logp, feasible),
                    'alpha': jnp.exp(log_alpha),
                    't': t,
                }
            except Exception as e:
                info = {
                    'qf_loss': qf_loss,
                    'qf': jax.nn.sigmoid(qf).mean(),
                    'q1_loss': q1_loss,
                    'q2_loss': q2_loss,
                    'q1': q1.mean(),
                    'q2': q2.mean(),
                    'policy_loss': policy_loss,
                    'feasible_policy_loss': policy_loss1.mean(),
                    'infeasible_policy_loss': policy_loss2.mean(),
                    'log_barrier': log_barrier.mean(),
                    'feasible_ratio': feasible.mean(),
                    # 'entropy': -masked_mean(new_logp, feasible),
                    'alpha': jnp.exp(log_alpha),
                    't': t,
                }
            return params, alg_state, info

        self.stateless_update = stateless_update
