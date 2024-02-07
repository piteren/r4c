from abc import abstractmethod, ABC
import numpy as np
from pypaq.pytypes import NUM
from typing import Dict, Any

from r4c.helpers import update_terminal_QVs
from r4c.actor import FiniTRActor


class QLearningActor(FiniTRActor, ABC):
    """ QLearningActor, supports finite actions space environments (FiniteActionsRLEnvy) """

    def __init__(
            self,
            gamma: float, # QLearning gamma (discount factor)
            **kwargs):
        FiniTRActor.__init__(self, **kwargs)
        self.gamma = gamma

    @abstractmethod
    def _get_QVs(self, observation:np.ndarray) -> np.ndarray:
        """ returns QVs (QV for all actions) for given observation """
        pass

    def get_QVs_batch(self, observations:np.ndarray) -> np.ndarray:
        """ returns QVs (for all actions) for given observations batch, here baseline implementation """
        return np.asarray([self._get_QVs(o) for o in observations])

    def _get_action(
            self,
            observation: np.ndarray,
            explore: bool=  False,
    ) -> NUM:
        """ returns action based on QVs """
        if explore:
            return int(np.random.choice(self.envy.num_actions()))
        qvs = self._get_QVs(observation)
        return int(np.argmax(qvs))

    @abstractmethod
    def _upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:
        """ updates QV for given observation and action, returns loss (TD Error - Temporal Difference Error?) """
        pass

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ extracts from a batch + adds new QV from Bellman Equation """

        next_observations_qvs = self.get_QVs_batch(batch['next_observations'])

        update_terminal_QVs(
            qvs=        next_observations_qvs,
            terminals=  batch['terminals'])

        new_qv = [
            r + self.gamma * max(no_qvs)
            for r, no_qvs in zip(batch['rewards'], next_observations_qvs)]

        return {
            'observations': batch['observations'],
            'actions':      batch['actions'],
            'new_qv':       np.asarray(new_qv)}

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates QV """
        loss = 0.0
        for obs, act, nqv in zip(training_data['observations'], training_data['actions'], training_data['new_qv']):
            loss += self._upd_QV(
                observation=    obs,
                action=         act,
                new_qv=         nqv)
        return {'loss': loss}

    def __str__(self):
        nfo = f'{super().__str__()}\n'
        nfo += f'> gamma: {self.gamma}'
        return nfo