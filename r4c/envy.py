from abc import abstractmethod, ABC
import numpy as np
from pypaq.pytypes import NUM
from pypaq.lipytools.pylogger import get_pylogger
from typing import List, Optional

from r4c.helpers import RLException



# base Environment interface
class Envy(ABC):
    """
    Such concept of Envy assumes that user (Actor) is responsible for
    resetting the Envy after reaching terminal state (e.g. before calling run()).
    """

    def __init__(
            self,
            seed: int,
            logger=     None,
            loglevel=   20):
        self._rlog = logger or get_pylogger(level=loglevel)
        self.seed = seed
        self._rlog.info(f'*** {self.__class__.__name__} (Envy) *** initialized')
        self._rlog.debug(self)

    # returns observation of current state
    @abstractmethod
    def get_observation(self) -> object: pass

    # plays action, goes to new state, may return something
    @abstractmethod
    def run(self, action:object) -> object: pass

    # is Envy currently in terminal state
    @abstractmethod
    def is_terminal(self) -> bool: pass

    # is Envy currently in terminal state + user won episode
    @abstractmethod
    def won(self) -> bool: pass

    # resets Envy to initial state, seed is managed by Envy
    def reset(self):
        self.reset_with_seed(seed=self.seed)
        self.seed += 1

    # resets Envy to initial state with given seed
    @abstractmethod
    def reset_with_seed(self, seed:int): pass

    # returns max number of steps in one episode, None means infinite
    @abstractmethod
    def get_max_steps(self) -> Optional[int]: pass

    def __str__(self):
        nfo = f'{self.__class__.__name__} (Envy)\n'
        nfo += f'> max steps: {self.get_max_steps()}'
        return nfo


# adds to Envy methods needed by base RL algorithms (used by Actor or Trainer)
class RLEnvy(Envy, ABC):

    # plays action, goes to new state, returns reward
    def run(self, action:NUM) -> float:
        """
        RLEnvy returns reward after each step (run).
        This value may (should?) be processed / overridden by Actor.
        Actor is supposed to train itself using information of reward
        that he defines / corrects while observing an Envy,
        he may apply discount, factor, moving average etc. to reward returned by Envy.
        (Actor does not need a reward to act with policy.)
        """
        pass

    # Envy current state rendering (for debug, preview etc.)
    def render(self): pass

    # prepares numpy vector of NUM from given observation object
    def observation_vector(self, observation:object) -> np.ndarray:
        """
        It may be implemented by RLEnvy, but is not mandatory,
        otherwise Actor should implement on itself since it is in fact Actor duty.
        Be careful about dtype and values, NN may not accept dtype==int.
        """
        raise RLException('RLEnvy not implemented observation_vector()')


# interface of RL Environment with finite actions number
class FiniteActionsRLEnvy(RLEnvy):

    def __init__(self, **kwargs):
        RLEnvy.__init__(self, **kwargs)

    # returns list of valid actions
    @abstractmethod
    def get_valid_actions(self) -> List[object]: pass

    # returns number of Envy actions
    def num_actions(self) -> int:
        return len(self.get_valid_actions())

    def __str__(self):
        nfo =  f'{super().__str__()}\n'
        nfo += f'> num_actions: {self.num_actions()}'
        return nfo