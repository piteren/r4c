from abc import abstractmethod, ABC
import numpy as np
from pypaq.pytypes import NUM
from pypaq.lipytools.pylogger import get_pylogger
from typing import List, Optional

from r4c.helpers import RLException


class Envy(ABC):
    """ Environment (abstract)
    this concept of Envy assumes that user (Actor) is responsible for
    resetting the Envy after reaching terminal state, e.g. before calling run() """

    def __init__(
            self,
            seed: int=  123,
            logger=     None,
            loglevel=   20,
    ):
        self._rlog = logger or get_pylogger(level=loglevel)
        self.seed = seed
        self._rlog.info(f'*** {self.__class__.__name__} (Envy) *** initialized')
        self._rlog.debug(self)

    @abstractmethod
    def get_observation(self) -> object:
        """ returns observation of current state """
        pass

    @abstractmethod
    def run(self, action:object) -> object:
        """ Envy plays (runs) an action, goes to new state, may return something """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """ Envy is in terminal state now"""
        pass

    @abstractmethod
    def has_won(self) -> bool:
        """ Envy is in terminal state now + user won episode """
        pass

    def reset(self):
        """ resets Envy to initial state """
        self.reset_with_seed(seed=self.seed)
        self.seed += 1

    @abstractmethod
    def reset_with_seed(self, seed:int):
        """ resets Envy to initial state with given seed """
        pass

    @property
    def max_steps(self) -> Optional[int]:
        """ returns max number of steps in one episode, None means infinite """
        raise NotImplementedError

    def __str__(self):
        nfo = f'{self.__class__.__name__} (Envy)\n'
        nfo += f'> max steps: {self.max_steps}'
        return nfo


class RLEnvy(Envy, ABC):
    """ adds to Envy methods needed by base RL algorithms (used by Actor or Trainer) """

    def run(self, action:NUM) -> float:
        """ plays action, goes to new state, returns reward
        RLEnvy returns reward after each step (run) """
        pass

    def render(self):
        """ Envy current state rendering, for debug, preview etc. """
        pass

    def observation_vector(self, observation:object) -> np.ndarray:
        """ prepares numpy vector of NUM from given observation object
        It may be implemented by RLEnvy, but is not mandatory,
        otherwise Actor should implement on itself since it is in fact Actor duty.
        Be careful about dtype and values, NN may not accept dtype==int. """
        raise RLException('RLEnvy not implemented observation_vector()')


class FiniteActionsRLEnvy(RLEnvy):
    """ interface of RL Environment with finite actions number """

    def __init__(self, **kwargs):
        RLEnvy.__init__(self, **kwargs)

    @abstractmethod
    def get_valid_actions(self) -> List[object]:
        """ returns list of valid actions """
        pass

    def num_actions(self) -> int:
        """ returns number of Envy actions """
        return len(self.get_valid_actions())

    def __str__(self):
        nfo =  f'{super().__str__()}\n'
        nfo += f'> num_actions: {self.num_actions()}'
        return nfo