import numpy as np
from pypaq.pytypes import NPL
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.plots import two_dim_multi
from typing import List, Union, Optional, Dict


# Reinforcement Learning Exception
class RLException(Exception): pass


# Experience Memory (for Actor)
class ExperienceMemory:
    """
    Experience Memory stores data (experience) generated
    while Actor plays with its policy on Envy.
    Data is stored as numpy arrays.
    """

    def __init__(
            self,
            max_size: Optional[int]=    None,
            seed: int=                  123):
        self._mem: Dict[str,np.ndarray] = {}
        self._init_mem()
        self.max_size = max_size
        np.random.seed(seed)

    def _init_mem(self):
        self._mem = {
            'observations':         None, # np.ndarray of NUM (2 dim)
            'actions':              None, # np.ndarray of NUM
            'rewards':              None, # np.ndarray of floats
            'next_observations':    None, # np.ndarray of NUM (2 dim)
            'terminals':            None, # np.ndarray of bool
            'wons':                 None} # np.ndarray of bool

    # adds given experience
    def add(self, experience:Dict[str, Union[NPL]]):

        # add or put
        for k in experience:
            ex_np = np.asarray(experience[k])
            if self._mem[k] is None:
                self._mem[k] = ex_np
            else:
                self._mem[k] = np.concatenate([self._mem[k], ex_np])

        # trim if needed
        if self.max_size and len(self) > self.max_size:
            for k in self._mem:
                self._mem[k] = self._mem[k][-self.max_size:]

    # returns random sample of non-duplicates from memory
    def get_sample(self, n:int) -> Dict[str,np.ndarray]:
        ixs = np.random.choice(len(self), n, replace=False)
        return {k: self._mem[k][ixs] for k in self._mem}

    # returns (copy) of full memory
    def get_all(self, reset=True) -> Dict[str,np.ndarray]:
        mc = {k: np.copy(self._mem[k]) for k in self._mem}
        if reset: self._init_mem()
        return mc

    def clear(self):
        self._init_mem()

    def __len__(self):
        return len(self._mem['observations']) if self._mem['observations'] is not None else 0


# normalizes x with zscore (0 mean 1 std), this is helpful for training, as rewards can vary considerably between episodes,
def zscore_norm(x:NPL):
    if len(x) < 2: return x
    return (x - np.mean(x)) / (np.std(x) + 0.00000001)

# prepares list of discounted accumulated return from [reward]
def discounted_return(
        rewards: List[float],
        discount: float,
) -> List[float]:
    dar = np.zeros_like(rewards, dtype=float)
    s = 0.0
    for i in reversed(range(len(rewards))):
        s = s * discount + rewards[i]
        dar[i] = s
    return list(dar)

# prepares list of moving_average return from rewards
def movavg_return(
        rewards: List[float],
        factor: float,           # (0.0-0.1> factor of current reward taken for update
) -> List[float]:
    mvr = np.zeros_like(rewards, dtype=float)
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

    if len(rewards) != len(terminals):
        raise RLException('len(rewards) should be equal to len(terminals)')

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
def plot_obs_act(observations:NPL, actions:NPL):

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
        movavg_factor: float=   0.1,
):

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
