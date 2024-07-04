from typing import Optional


def make_env(id: str, max_episode_steps: Optional[int] = None):
    if 'Point' in id or 'Car' in id:
        from safety_gym_extension.utils.make import make_env
        return make_env(id)
    else:
        from simple_safe_env.utils.make import make_env
        return make_env(id, max_episode_steps=max_episode_steps)


def make_model(id: str):
    from simple_safe_env.utils.make import make_model
    return make_model(id)
