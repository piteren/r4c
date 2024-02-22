import numpy as np
from pypaq.pytypes import NPL
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.plots import two_dim_multi
from typing import List, Optional


class R4Cexception(Exception):
    pass


def zscore_norm(x:NPL) -> np.ndarray:
    """ normalizes x with zscore (0 mean 1 std)
    it is helpful for training, as rewards can vary considerably between episodes """

    if type(x) is not np.ndarray:
        x = np.asarray(x)

    if len(x) < 2: return x
    return (x - np.mean(x)) / (np.std(x) + 0.00000001)


def da_return(reward:NPL, discount:float) -> np.ndarray:
    """ prepares discounted accumulated reward """
    dar = np.zeros_like(reward, dtype=float)
    s = 0.0
    for i in reversed(range(len(reward))):
        s = s * discount + reward[i]
        dar[i] = s
    return dar


def update_terminal_values(value:np.ndarray, terminal:np.ndarray) -> None:
    """ sets terminal states of value to zeroes (in place) """
    qvs_terminal = np.zeros_like(value[0])
    for ix, t in enumerate(terminal):
        if t:
            value[ix] = qvs_terminal


def split_reward(reward:NPL, terminal:NPL) -> List[np.ndarray]:
    """ splits reward into episode reward """

    if len(reward) != len(terminal):
        raise R4Cexception('len(reward) should be equal to len(terminal)')

    episode_reward = []
    cep = []
    for r, t in zip(reward, terminal):
        cep.append(r)
        if t:
            episode_reward.append(np.asarray(cep))
            cep = []
    if cep:
        episode_reward.append(np.asarray(cep))
    return episode_reward


def plot_obs_act(observation:NPL, action:NPL):
    """ plots batch of observation and action """

    if type(observation) is not np.ndarray:
        observation = np.asarray(observation)

    oL = np.split(observation, observation.shape[-1], axis=-1)
    data = oL + [action]
    two_dim_multi(
        ys=     data,
        names=  [f'obs_{ix}' for ix in range(len(oL))] + ['action'])


def plot_reward(
        reward,
        terminal: Optional[NPL]=    None,
        discount: float=            0.9,
):
    """ plots batch of reward and some variants """

    ys = [reward]
    names = ['reward']

    if terminal is not None:
        da_ret = []
        for rs in split_reward(reward, terminal):
            da_ret += da_return(reward=rs, discount=discount)
        da_ret_znorm = zscore_norm(da_ret)

        ys += [da_ret, da_ret_znorm]
        names += ['da_returns', 'da_returns_znorm']

    two_dim_multi(ys=ys, names=names, legend_loc='lower left')
