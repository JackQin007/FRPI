import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def find_max_step(path: str) -> int:
    max_step = 0
    prefix = 'params_'
    suffix = '.pkl'
    for fn in os.listdir(path):
        if fn.startswith(prefix) and fn.endswith(suffix):
            max_step = max(max_step, int(fn[len(prefix):-len(suffix)]))
    return max_step


def find_all_steps(path: str) -> int:
    steps = []
    prefix = 'params_'
    suffix = '.pkl'
    for fn in os.listdir(path):
        if fn.startswith(prefix) and fn.endswith(suffix):
            steps.append(int(fn[len(prefix):-len(suffix)]))
    return sorted(steps)
