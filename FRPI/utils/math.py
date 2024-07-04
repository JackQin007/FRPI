import math
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import linear_algebra


def masked_mean(
    x: jnp.ndarray,
    mask: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> jnp.ndarray:
    return (mask * x).sum(axis) / jnp.maximum(mask.sum(axis), 1)


def clip_grad_norm(grad: base.Updates, max_norm: float) -> base.Updates:
    g_norm = linear_algebra.global_norm(grad)
    g_norm = jnp.maximum(max_norm, g_norm)
    grad = jax.tree_util.tree_map(lambda t: (t / g_norm) * max_norm, grad)
    return grad


def discounted_sum(
    x: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    dc_sum = last_value
    for i in reversed(range(x.shape[0])):
        dc_sum = x[i] + mask[i] * gamma * dc_sum
    return dc_sum


def discounted_sum_stacked(
    x: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    dc_sum = [last_value]
    for i in reversed(range(x.shape[0])):
        dc_sum.append(x[i] + mask[i] * gamma * dc_sum[-1])
    dc_sum = jnp.stack(dc_sum[:0:-1])
    return dc_sum


def angle_normalize(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi
