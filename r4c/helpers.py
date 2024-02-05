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


def da_returns(rewards:List[float], discount:float) -> List[float]:
    """ prepares list of returns <- discounted accumulated rewards """
    dar = np.zeros_like(rewards, dtype=float)
    s = 0.0
    for i in reversed(range(len(rewards))):
        s = s * discount + rewards[i]
        dar[i] = s
    return list(dar)


def update_terminal_QVs(qvs:np.ndarray, terminals:np.ndarray) -> None:
    """ sets terminal states of QVs to zeroes (in place) """
    qvs_terminal = np.zeros_like(qvs[0])
    for ix, t in enumerate(terminals):
        if t: qvs[ix] = qvs_terminal


def split_rewards(rewards, terminals) -> List[List[float]]:
    """ splits rewards into episode rewards """

    if len(rewards) != len(terminals):
        raise R4Cexception('len(rewards) should be equal to len(terminals)')

    episode_rewards = []
    cep = []
    for r, t in zip(rewards, terminals):
        cep.append(r)
        if t:
            episode_rewards.append(cep)
            cep = []
    if cep: episode_rewards.append(cep)
    return episode_rewards


def plot_obs_act(observations:NPL, actions:NPL):
    """ plots batch of observations and actions """

    if type(observations) is not np.ndarray:
        observations = np.asarray(observations)

    oL = np.split(observations, observations.shape[-1], axis=-1)
    data = oL + [actions]
    two_dim_multi(
        ys=     data,
        names=  [f'obs_{ix}' for ix in range(len(oL))] + ['actions'])


def plot_rewards(
        rewards,
        terminals: Optional=    None,
        discount: float=        0.9,
):
    """ plots batch of rewards and some variants """

    ys = [rewards]
    names = ['rewards']

    if terminals is not None:
        da_ret = []
        for rs in split_rewards(rewards, terminals):
            da_ret += da_returns(rewards=rs, discount=discount)
        da_ret_znorm = zscore_norm(da_ret)

        ys += [da_ret, da_ret_znorm]
        names += ['da_returns', 'da_returns_znorm']

    two_dim_multi(ys=ys, names=names, legend_loc='lower left')
