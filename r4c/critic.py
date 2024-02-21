from abc import ABC
from pypaq.lipytools.printout import stamp
from typing import Optional

from r4c.actor import Actor, TrainableActor


class Critic(ABC):
    """ Critic (abstract) """

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

    # TODO:
    """
    @abstractmethod
    def save(self): pass

    @abstractmethod
    def load(self): pass

    def __str__(self):
        nfo =  f'{self.__class__.__name__} (Critic) : {self.name}\n'
        nfo += f'> observation width: {self._observation_vector(self.envy.get_observation()).shape[-1]}'
        return nfo
    """

class TrainableCritic(Critic, ABC):

    def __init__(self, actor:TrainableActor, **kwargs):
        Critic.__init__(self, actor=actor, **kwargs)
        self.actor = actor # just for typing