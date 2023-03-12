import numpy as np
from pypaq.lipytools.moving_average import MovAvg
from typing import List, Union

NUM = Union[int,float]


# Reinforcement Learning Exception
class RLException(Exception): pass


# normalizes x with zscore (0 mean 1 std), this is helpful for training, as rewards can vary considerably between episodes,
def zscore_norm(x):
    if len(x) < 2: return x
    return (x - np.mean(x)) / (np.std(x) + 0.00000001)

# prepares list of discounted accumulated return from [reward]
def discounted_return(
        rewards: List[float],
        discount: float) -> List[float]:
    dar = np.zeros_like(rewards)
    s = 0.0
    for i in reversed(range(len(rewards))):
        s = s * discount + rewards[i]
        dar[i] = s
    return list(dar)

# prepares list of moving_average return from rewards
def movavg_return(
        rewards: List[float],
        factor: float           # (0.0-0.1> factor of current reward taken for update
) -> List[float]:
    mvr = np.zeros_like(rewards)
    s = rewards[-1]
    mavg = MovAvg(factor=factor, first_avg=False)
    mvr[-1] = mavg.upd(s)
    for i in reversed(range(len(rewards[:-1]))):
        mvr[i] = mavg.upd(rewards[i])
    return list(mvr)

# sets terminal states of QVs to zeroes (in place)
def update_terminal_QVs(qvs:np.ndarray, terminals:np.ndarray) -> None:
    qvs_terminal = np.zeros_like(qvs[0])
    for ix, t in enumerate(terminals):
        if t: qvs[ix] = qvs_terminal