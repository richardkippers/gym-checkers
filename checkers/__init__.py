from .game import Checkers
from gym.envs.registration import register

__all__ = [Checkers]

register(
    id='Checkers-v0',
    entry_point='checkers.game:Checkers'
)
