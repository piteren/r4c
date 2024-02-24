from abc import ABC, abstractmethod
import numpy as np
from pypaq.pytypes import NUM
from pypaq.pms.base import POINT
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.pylogger import get_child
from torchness.motorch import MOTorch, Module
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import Optional, Dict, Any

from r4c.actor import Actor, TrainableActor, FiniTRActor


class Critic(ABC):
    """ Critic does not exist without an Actor,
    it gets many properties from an Actor """

    def __init__(self, actor:Actor, name:Optional[str]=None):

        self.actor = actor

        if name is None:
            name = f'{self.__class__.__name__}'
        if actor.add_stamp:
            name += f'_{stamp()}'
        self.name = name

        self.save_dir = f'{self.actor.save_dir}/{self.name}'

        self.logger = self.actor.logger
        self.logger.info(f'*** {self.__class__.__name__} (Critic) : {self.name} *** initializes..')

    @abstractmethod
    def get_value(self, observation:np.ndarray) -> NUM:
        """ Critic returns a value for state / actions based on observation """
        pass

    @abstractmethod
    def save(self): pass

    @abstractmethod
    def load(self): pass

    def __str__(self):
        return f'{self.__class__.__name__} (Critic) : {self.name}'


class TrainableCritic(Critic, ABC):
    """ TrainableCritic learns (value?) from an Envy """

    def __init__(self, actor:TrainableActor, **kwargs):
        super().__init__(actor=actor, **kwargs)
        self.actor = actor # just for typing

    @abstractmethod
    def build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ prepares Critic training data, usually called by an Actor """
        pass

    @abstractmethod
    def update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates Critic (value function) returns metrics with 'loss', usually called by an Actor """
        pass

    def publish(self, metrics:Dict[str,Any]) -> None:
        """ usually called by an Actor """
        if self.actor.tbwr:
            for k, v in metrics.items():
                self.actor.tbwr.add(value=v, tag=f'critic/{k[7:]}', step=self.actor.upd_step)


class FiniTRCritic(TrainableCritic, ABC):
    """ FiniTRCritic is a TrainableCritic for FiniteActionsRLEnvy """

    def __init__(self, actor:FiniTRActor, **kwargs):
        TrainableCritic.__init__(self, actor=actor, **kwargs)
        self.actor = actor # just for typing


class MOTRCritic(TrainableCritic, ABC):
    """ MOTRCritic is a MOTorch (NN model) based TrainableCritic
    its functionality is similar to MOTRActor
    common similarity could be extracted with OOP (I tried but failed) """

    def __init__(
            self,
            module_type: Optional[type(Module)],
            model_type: type(MOTorch)=      MOTorch,
            motorch_point: Optional[POINT]= None,
            **kwargs):

        super().__init__(**kwargs)

        self.model = model_type(
            module_type=        module_type,
            **self._critic_motorch_point(),
            **(motorch_point or {}))

        self._zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       self.actor.tbwr) if self.actor.tbwr else None

    def _critic_motorch_point(self) -> POINT:
        """ prepares Critic specific motorch_point addon """
        return {
            'name':                 self.name,
            'observation_width':    self.actor.observation_width,
            'discount':             self.actor.discount,
            'seed':                 self.actor.seed,
            'logger':               get_child(self.logger),
            'hpmser_mode':          self.actor.hpmser_mode}

    def get_value(self, observation:np.ndarray) -> np.ndarray:
        """ implements value from NN model """
        return self.model(observation=observation)['value'].detach().cpu().numpy()

    def update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ update with NN backprop """
        return self.model.backward(**training_data)

    def publish(self, metrics:Dict[str,Any]) -> None:
        if self.actor.tbwr:
            metrics.pop('critic_value')
            self._zepro.process(zeroes=metrics.pop('critic_zeroes'), step=self.actor.upd_step)
            super().publish(metrics={k[7:]: metrics[k] for k in metrics})

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()

    def __str__(self) -> str:
        nfo = f'{super().__str__()}\n'
        nfo += str(self.model)
        return nfo