from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

from .lbforaging import ForagingEnv
# from .join1 import Join1Env
from .predator_prey import PredatorPreyEnv
from .hallway import JoinNEnv
from .hallway import Join1Env

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "foraging": partial(env_fn, env=ForagingEnv),
    # "join1": partial(env_fn, env=Join1Env),
    "pp": partial(env_fn, env=PredatorPreyEnv),
    "hallway_group": partial(env_fn, env=JoinNEnv),
    "hallway": partial(env_fn, env=Join1Env)
}
