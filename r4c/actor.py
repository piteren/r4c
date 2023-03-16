from abc import abstractmethod, ABC
import numpy as np
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.pylogger import get_pylogger
from torchness.tbwr import TBwr
from typing import Optional, Dict, Any

from r4c.envy import RLEnvy
from r4c.helpers import RLException, NUM, plot_obs_act, plot_rewards


# just abstract Actor
class Actor(ABC):

    # returns Actor action based on observation according to Actor policy
    @abstractmethod
    def get_action(self, observation:object) -> object: pass


# Plays & Learns on Envy
class TrainableActor(Actor, ABC):
    """
    Trainable Actor covers much more than an Actor known from RL theory,
    may consist of many subcomponents like Actor, Critic, QTable, etc.
    """

    def __init__(
            self,
            envy: RLEnvy,
            name: Optional[str]=    None,
            add_stamp: bool=        True,
            save_topdir: str=       '_models',
            logger: Optional=       None,
            loglevel: int=          20,
            publish_TB: bool=       True,
            hpmser_mode: bool=      False,
            seed: int=              123):

        self.envy = envy

        if name is None:
            name = f'TrainableActor_{self.envy.__class__.__name__}'
        if add_stamp: name += f'_{stamp()}'
        self.name = name

        self.save_topdir = save_topdir

        self._rlog = logger or get_pylogger(
            folder= self.get_save_dir(),
            level=  loglevel)

        # early override
        if hpmser_mode:
            publish_TB = False

        self.seed = seed
        np.random.seed(self.seed)

        self._tbwr = TBwr(logdir=self.get_save_dir()) if publish_TB else None
        self._upd_step = 0  # global update step

        self._rlog.info(f'*** {self.__class__.__name__} (TrainableActor) : {self.name} *** initialized')
        self._rlog.debug(self)

    # prepares numpy vector from observation in type accepted by self, first tries to get from RLEnvy
    def observation_vector(self, observation:object) -> np.ndarray:
        try:
            return self.envy.observation_vector(observation)
        except RLException:
            raise RLException ('TrainableActor should implement get_observation_vec()')

    # specifies observation and return type, adds exploration & sampling
    @abstractmethod
    def get_action(
            self,
            observation: np.ndarray,
            explore: bool=  False, # returns exploring action
            sample: bool=   False, # samples action (from probability?), option which may be helpful for training
    ) -> NUM: pass

    # wraps data preparation + update + publish, returns loss
    def update_with_experience(self, batch:Dict[str,np.ndarray], inspect:bool) -> float:
        training_data = self._build_training_data(batch=batch)
        metrics = self._update(training_data=training_data)
        self._publish(batch=batch, training_data=training_data, metrics=metrics, inspect=inspect)
        self._upd_step += 1
        return metrics['loss']

    # extracts data from a batch + eventually adds new
    @abstractmethod
    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]: pass

    # updates policy or value function (Actor, Critic ar any other component), returns metrics with loss
    @abstractmethod
    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]: pass

    # publishes to TB / inspects data in research mode, here baseline mode
    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            training_data: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
            inspect: bool,
    ) -> None:

        if self._tbwr:
            self._tbwr.add(value=metrics['loss'], tag=f'actor/loss', step=self._upd_step)

        if inspect:
            plot_obs_act(observations=batch['observations'], actions=batch['actions'])
            plot_rewards(rewards=batch['rewards'])

    # returns Actor save directory
    def get_save_dir(self) -> str:
        return f'{self.save_topdir}/{self.name}'

    # saves Actor (self)
    @abstractmethod
    def save(self): pass

    # loads Actor (self)
    @abstractmethod
    def load(self): pass

    # returns some info about Actor
    def __str__(self):
        nfo =  f'{self.__class__.__name__} (TrainableActor) : {self.name}\n'
        nfo += f'> observation width: {self.observation_vector(self.envy.get_observation()).shape[-1]}'
        return nfo