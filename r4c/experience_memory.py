import numpy as np
from pypaq.lipytools.pylogger import get_pylogger
from typing import Optional, Dict

from r4c.helpers import R4Cexception


class ExperienceMemory:
    """ Experience Memory (for TrainableActor)
    Experience Memory stores data (experience) generated
    while TrainableActor plays with its policy on Envy """

    def __init__(
            self,
            max_size: Optional[int]=    None,
            seed: int=                  123,
            logger: Optional =          None,
            loglevel: int =             20):

        if not logger:
            logger = get_pylogger(level=loglevel)
        self.logger = logger

        self._mem: Dict[str,np.ndarray] = {}
        self.max_size = max_size
        np.random.seed(seed)

        self.logger.info(f'*** {self.__class__.__name__} *** initialized, size: {self.max_size}')

    def add(self, experience:Dict[str,np.ndarray]):
        """ adds given experience
        it is up to Actor decision whether it is ok to add more experience to actual memory
        probably usually Actor should add new experience to empty memory """

        # add or put
        for k in experience:
            ed = experience[k]
            if k not in self._mem: self._mem[k] = ed
            else:                  self._mem[k] = np.concatenate([self._mem[k], ed])

        # trim from the beginning, if needed
        if self.max_size and len(self) > self.max_size:
            for k in self._mem:
                self._mem[k] = self._mem[k][-self.max_size:]

    def get_sample(self, n:int) -> Dict[str,np.ndarray]:
        """ returns random sample of non-duplicates from memory """
        if n > len(self):
            raise R4Cexception(f'cannot sample {n} samples from memory of size {len(self)}')
        ixs = np.random.choice(len(self), n, replace=False)
        return {k: self._mem[k][ixs] for k in self._mem}

    def get_all(self, reset=True) -> Dict[str,np.ndarray]:
        """ returns all memory data """
        mc = {k: np.copy(self._mem[k] if not reset else self._mem[k]) for k in self._mem}
        if reset:
            self.reset()
        return mc

    def reset(self):
        self._mem = {}

    def __len__(self):
        return len(self._mem['observation']) if self._mem['observation'] is not None else 0

    def __str__(self):
        s = f'{self.__class__.__name__}, size: {self.max_size}'
        if self._mem:
            keys = list(self._mem.keys())
            size = len(self._mem[keys[0]])
            s += f'\ngot ({len(keys)}) keys: {", ".join(keys)}'
            s += f'\nactual size: {size}'
            for k in keys:
                s += f'\n> key: {k} --- {self._mem[k].shape} {self._mem[k].dtype}'
