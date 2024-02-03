import numpy as np
from pypaq.pytypes import NPL
from typing import Optional, Dict, Union


class ExperienceMemory:
    """ Experience Memory (for TrainableActor)
    Experience Memory stores data (experience) generated
    while TrainableActor plays with its policy on Envy """

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

    def add(self, experience:Dict[str, Union[NPL]]):
        """ adds given experience """

        # add or put
        for k in experience:
            ex_np = np.asarray(experience[k])
            if self._mem[k] is None:
                self._mem[k] = ex_np
            else:
                self._mem[k] = np.concatenate([self._mem[k], ex_np])

        # trim from the beginning, if needed
        if self.max_size and len(self) > self.max_size:
            for k in self._mem:
                self._mem[k] = self._mem[k][-self.max_size:]

    def get_sample(self, n:int) -> Dict[str,np.ndarray]:
        """ returns random sample of non-duplicates from memory """
        ixs = np.random.choice(len(self), n, replace=False)
        return {k: self._mem[k][ixs] for k in self._mem}

    def get_all(self, reset=True) -> Dict[str,np.ndarray]:
        """ returns all memory data """
        mc = {k: np.copy(self._mem[k] if not reset else self._mem[k]) for k in self._mem}
        if reset:
            self._init_mem()
        return mc

    def clear(self):
        self._init_mem()

    def __len__(self):
        return len(self._mem['observations']) if self._mem['observations'] is not None else 0
