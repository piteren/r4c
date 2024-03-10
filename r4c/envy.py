from abc import abstractmethod, ABC
import numpy as np
from pypaq.pytypes import NUM
from pypaq.lipytools.pylogger import get_pylogger
from typing import List, Optional

from r4c.helpers import R4Cexception


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
        self.logger = logger or get_pylogger(level=loglevel)
        self.logger.info(f'*** {self.__class__.__name__} (Envy) *** initializes..')
        self.seed = seed

        self.state = None
        self.reset()

    @property
    def observation(self) -> object:
        """ returns observation of current state """
        return self.state

    @abstractmethod
    def sample_action(self) -> NUM:
        """ returns random action """
        pass

    @abstractmethod
    def run(self, action:NUM) -> object:
        """ Envy plays (runs) an action, goes to new state -> should update self.state, may return something """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """ Envy is in terminal state now """
        pass

    @abstractmethod
    def has_won(self) -> bool:
        """ Envy is in terminal state now + user won episode,
        this state may be never True for some Envies. """
        pass

    def reset(self) -> object:
        """ resets Envy to initial state, returns state """
        self.state = self.reset_with_seed(seed=self.seed)
        self.seed += 1
        return self.state

    @abstractmethod
    def reset_with_seed(self, seed:int) -> object:
        """ resets Envy to initial state with given seed, returns state """
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
    """ RLEnvy - adds some RL methods to Envy """

    @abstractmethod
    def build_renderable(self) -> "RLEnvy":
        """ returns a duplicate of self that is renderable """
        pass

    def run(self, action:NUM) -> float:
        """ plays action, goes to new state, returns reward
        RLEnvy returns reward after each step (run) """
        pass

    def observation_vector(self, observation:object) -> np.ndarray:
        """ prepares numpy vector of NUM from given observation object
        It may be implemented by RLEnvy, but is not mandatory,
        otherwise Actor should implement on itself since it is in fact Actor duty.
        Be careful about dtype and values, NN may not accept dtype==int. """
        raise R4Cexception('RLEnvy not implemented observation_vector()')


class FiniteActionsRLEnvy(RLEnvy, ABC):
    """ FiniteActionsRLEnvy - RLEnvy with finite action space """

    @abstractmethod
    def get_valid_actions(self) -> List[object]:
        """ returns list of valid actions """
        pass

    @property
    def num_actions(self) -> int:
        """ returns number of Envy actions """
        return len(self.get_valid_actions())

    def __str__(self):
        nfo =  f'{super().__str__()}\n'
        nfo += f'> num_actions: {self.num_actions}'
        return nfo


class CASRLEnvy(RLEnvy, ABC):
    """ CASRLEnvy - RLEnvy with continuous action space """

    @property
    @abstractmethod
    def action_width(self) -> int:
        """ returns width of action """
        pass

    def __str__(self):
        nfo =  f'{super().__str__()}\n'
        nfo += f'> action_width: {self.action_width}'
        return nfo