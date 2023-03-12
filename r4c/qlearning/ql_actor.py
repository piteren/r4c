from abc import abstractmethod, ABC
import numpy as np
from pypaq.lipytools.softmax import softmax
from typing import Dict, Any

from r4c.actor import TrainableActor
from r4c.envy import FiniteActionsRLEnvy


# QLearningActor, supports finite actions space environments (FiniteActionsRLEnvy)
class QLearningActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            seed: int,
            **kwargs):

        TrainableActor.__init__(self, envy=envy, **kwargs)
        self._envy = envy  # to update type (for pycharm only)

        np.random.seed(seed)

        self._rlog.info('*** QLearningActor *** initialized')
        self._rlog.info(f'> num_actions: {self._envy.num_actions()}')
        self._rlog.info(f'> seed:        {seed}')

    # returns QVs (QV for all actions) for given observation
    @abstractmethod
    def _get_QVs(self, observation:np.ndarray) -> np.ndarray: pass

    # returns QVs (for all actions) for given observations batch, here baseline implementation
    def get_QVs_batch(self, observations:np.ndarray) -> np.ndarray:
        return np.asarray([self._get_QVs(o) for o in observations])

    # returns action based on QVs
    def get_policy_action(
            self,
            observation: np.ndarray,
            sampled=    False,  # (experimental) whether to sample action with softmax on QVs
    ) -> int:

        qvs = self._get_QVs(observation)

        if sampled:
            obs_probs = softmax(qvs)
            return np.random.choice(len(qvs), p=obs_probs)

        return int(np.argmax(qvs))

    # updates QV for given observation and action, returns loss
    @abstractmethod
    def _upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float: pass

    # updates QV
    def update_with_experience(
            self,
            batch: Dict[str,np.ndarray],
            inspect: bool,
    ) -> Dict[str, Any]:
        loss = 0.0
        for obs, act, nqv in zip(batch['observations'], batch['actions'], batch['new_qvs']):
            loss += self._upd_QV(
                observation=    obs,
                action=         act,
                new_qv=         nqv)
        return {'loss': loss}