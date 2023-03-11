from abc import abstractmethod, ABC
import numpy as np
from typing import Optional, Dict, Any, List, Union

from pypaq.lipytools.printout import stamp
from pypaq.lipytools.pylogger import get_pylogger
from r4c.envy import RLEnvy
from r4c.helpers import RLException, NUM


# just Actor
class Actor(ABC):

    # returns Actor action based on observation according to Actor policy
    @abstractmethod
    def get_policy_action(self, observation: object) -> object: pass


# cooperates with RLTrainer, prepares obs_vec, updates self policy, saves
class TrainableActor(Actor, ABC):

    def __init__(
            self,
            envy: RLEnvy,
            name: str=              'TrainableActor',
            name_timestamp: bool=   True,
            logger: Optional=       None,
            loglevel: int=          20,
            **kwargs):

        if name_timestamp: name += f'_{stamp()}'
        self.name = name
        self._envy = envy

        self._rlog = logger or get_pylogger(level=loglevel)
        self._rlog.info(f'*** TrainableActor *** initialized')
        self._rlog.info(f'> name:              {self.name}')
        self._rlog.info(f'> Envy:              {self._envy.__class__.__name__}')
        self._rlog.info(f'> observation width: {self.get_observation_vec(self._envy.get_observation()).shape[-1]}')
        self._rlog.info(f'> not used kwargs:   {kwargs}')

    # prepares numpy vector from observation, first tries to get from RLEnvy
    def get_observation_vec(self, observation: object) -> np.ndarray:
        try:
            return self._envy.prep_observation_vec(observation)
        except RLException:
            raise RLException ('TrainableActor should implement get_observation_vec()')

    # adds sampling (from probability?) option which may be helpful for training, returns number
    @abstractmethod
    def get_policy_action(self, observation:object, sampled=False) -> NUM: pass

    # updates policy
    @abstractmethod
    def update_with_experience(
            self,
            batch: Dict[str,np.ndarray],
            inspect: bool,
    ) -> Dict[str,Any]:
        """
        updates (self) policy with 'batch' of experience data
        returns dict with some update "metrics" like a loss etc.
        those metrics may be used by RLTrainer for publish, monitoring, training process update..
        currently supported:
        - any number (float?) will be published to TB (with key from dict and Trainer._upd_step)
        - 'probs' Tensor will be published to TB with avg_probs()
        - 'zeroes' will be published by Trainer._zepro
        other have to be removed and may be eventually inspected by an Actor
        """
        pass

    @abstractmethod
    # returns Actor TOP save directory
    def _get_save_topdir(self) -> str: pass

    # returns Actor save directory
    def get_save_dir(self) -> str:
        return f'{self._get_save_topdir()}/{self.name}'

    # saves Actor (self)
    @abstractmethod
    def save(self): pass

    # returns some info about Actor
    @abstractmethod
    def __str__(self) -> str: pass