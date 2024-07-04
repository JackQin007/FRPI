from typing import Callable, Optional

import gymnasium as gym
import numpy as np


def evaluate_one_episode(
    env: gym.Env,
    get_action: Callable[[np.ndarray], np.ndarray],
    seed: Optional[int] = None,
    options: Optional[dict] = None,
) -> dict:
    obs, info = env.reset(seed=seed, options=options)
    ep_ret = 0.0
    ep_cost = info.get('cost', 0.0)
    ep_len = 0
    obss = [obs]
    actions = []
    rewards = []
    while True:
        action = get_action(obs)
        obs, reward, terminated, truncated, next_info = env.step(action)
        ep_ret += reward
        ep_cost += next_info.get('cost', 0.0)
        ep_len += 1
        obss.append(obs)
        actions.append(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return {
        'episode_return': ep_ret,
        'episode_cost': ep_cost,
        'episode_length': ep_len,
        'obs': np.stack(obss),
        'action': np.stack(actions),
        'reward': np.stack(rewards)
    }
