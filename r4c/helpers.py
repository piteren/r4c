import numpy as np
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.plots import two_dim_multi
from typing import List, Union, Optional

NUM = Union[int, float, np.ndarray]


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

# splits rewards into episode rewards
def split_rewards(rewards, terminals) -> List[List[float]]:
    episode_rewards = []
    cep = []
    for r, t in zip(rewards, terminals):
        cep.append(r)
        if t:
            episode_rewards.append(cep)
            cep = []
    if cep: episode_rewards.append(cep)
    return episode_rewards

# plots (inspects) batch of observations and actions
def plot_obs_act(
        observations,
        actions):

    if type(observations) is not np.ndarray:
        observations = np.asarray(observations)

    oL = np.split(observations, observations.shape[-1], axis=-1)
    data = oL + [actions]
    two_dim_multi(
        ys=     data,
        names=  [f'obs_{ix}' for ix in range(len(oL))] + ['actions'])

# plots (inspects) batch of rewards
def plot_rewards(
        rewards,
        terminals: Optional=    None,
        discount: float=        0.9,
        movavg_factor: float=   0.1):

    ys = [rewards]
    names = ['rewards']

    if terminals is not None:
        dret_disc = []
        dret_mavg = []
        for rs in split_rewards(rewards, terminals):
            dret_disc += discounted_return(rewards=rs, discount=discount)
            dret_mavg += movavg_return(rewards=rs, factor=movavg_factor)
        dret_disc_norm = zscore_norm(dret_disc)
        dret_mavg_norm = zscore_norm(dret_mavg)

        ys += [dret_disc, dret_mavg, dret_disc_norm, dret_mavg_norm]
        names += ['dret_disc', 'dret_mavg', 'dret_disc_norm', 'dret_mavg_norm']

    two_dim_multi(ys=ys, names=names, legend_loc='lower left')
