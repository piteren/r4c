from abc import abstractmethod, ABC
import numpy as np
from pypaq.lipytools.softmax import softmax
from typing import Dict, Any

from r4c.helpers import NUM, update_terminal_QVs
from r4c.actor import TrainableActor
from r4c.envy import FiniteActionsRLEnvy


# QLearningActor, supports finite actions space environments (FiniteActionsRLEnvy)
class QLearningActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            gamma: float,               # QLearning gamma (discount factor)
            **kwargs):
        TrainableActor.__init__(self, envy=envy, **kwargs)
        self.envy = envy  # to update type (for pycharm only)
        self.gamma = gamma

    # returns QVs (QV for all actions) for given observation
    @abstractmethod
    def _get_QVs(self, observation:np.ndarray) -> np.ndarray: pass

    # returns QVs (for all actions) for given observations batch, here baseline implementation
    def get_QVs_batch(self, observations:np.ndarray) -> np.ndarray:
        return np.asarray([self._get_QVs(o) for o in observations])

    # returns action based on QVs
    def get_action(
            self,
            observation: np.ndarray,
            explore: bool=  False,
            sample: bool=   False,  # (experimental) whether to sample action with softmax on QVs
    ) -> NUM:

        if explore:
            return int(np.random.choice(self.envy.num_actions()))

        qvs = self._get_QVs(observation)

        if sample:
            obs_probs = softmax(qvs)
            return np.random.choice(len(qvs), p=obs_probs)

        return np.argmax(qvs)

    # updates QV for given observation and action, returns loss (TD Error - Temporal Difference Error?)
    @abstractmethod
    def _upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float: pass

    # extracts from a batch + adds new QV from Bellman Equation
    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:

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

    # updates QV
    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        loss = 0.0
        for obs, act, nqv in zip(training_data['observations'], training_data['actions'], training_data['new_qv']):
            loss += self._upd_QV(
                observation=    obs,
                action=         act,
                new_qv=         nqv)
        return {'loss': loss}

    # publishes loss
    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            training_data: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
    ) -> None:
        if self._tbwr:
            self._tbwr.add(value=metrics['loss'], tag=f'actor/loss', step=self._upd_step)


    def __str__(self):
        nfo = f'{super().__str__()}\n'
        nfo += f'> gamma: {self.gamma}'
        return nfo