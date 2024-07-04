from typing import Sequence

import jax

from FRPI.agent.base import Agent

from FRPI.agent.frpi import FRPIAgent

from FRPI.agent.frpi_sac import FRPISACAgent



def make_agent(
    alg: str,
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
) -> Agent:

    if alg == 'frpi':
        agent = FRPIAgent(key, obs_dim, act_dim, hidden_sizes)
    elif alg == 'frpi-sac':
        agent = FRPISACAgent(key, obs_dim, act_dim, hidden_sizes)
    else:
        ValueError(f'Invalid algorithm {alg}!')
    return agent
