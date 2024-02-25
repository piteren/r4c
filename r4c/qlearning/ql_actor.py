from abc import abstractmethod, ABC
import numpy as np
from pypaq.pytypes import NUM
from typing import Dict, Any

from r4c.helpers import update_terminal_values
from r4c.actor import FiniTRActor


class QLearningActor(FiniTRActor, ABC):
    """ QLearningActor, supports finite actions space """

    @abstractmethod
    def _get_QVs(self, observation:np.ndarray) -> np.ndarray:
        """ returns QVs (QV for all actions) for given observation """
        pass

    def get_QVs_batch(self, observation:np.ndarray) -> np.ndarray:
        """ returns QVs (for all actions) for given observation batch, here baseline implementation """
        return np.asarray([self._get_QVs(o) for o in observation])

    def get_action(self, observation:np.ndarray) -> Dict[str,NUM]:
        """ returns action based on QVs -> action with a max QV """
        qvs = self._get_QVs(observation)
        return {'action': int(np.argmax(qvs))}

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ extracts from a batch + adds new QV from Bellman Equation """

        dk = ['observation', 'action']
        training_data = {k: batch[k] for k in dk}

        next_observation_qvs = self.get_QVs_batch(batch['next_observation'])
        update_terminal_values(value=next_observation_qvs, terminal=batch['terminal'])
        new_qv = [
            r + self.discount * max(no_qvs) # Bellman equation
            for r, no_qvs in zip(batch['reward'], next_observation_qvs)]
        training_data['new_qv'] = np.asarray(new_qv)

        return training_data