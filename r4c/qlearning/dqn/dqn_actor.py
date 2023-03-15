from abc import ABC
import numpy as np
from torchness.motorch import MOTorch, Module
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import Optional, Dict, Any

from r4c.helpers import RLException
from r4c.qlearning.ql_actor import QLearningActor
from r4c.qlearning.dqn.dqn_actor_module import DQNModel


# DQN (NN based) QLearningActor
class DQNActor(QLearningActor, ABC):

    def __init__(
            self,
            name: str=                              'DQNActor',
            module_type: Optional[type(Module)]=    DQNModel,
            **kwargs):

        QLearningActor.__init__(
            self,
            name=   name,
            **kwargs)

        # some overrides and updates
        kwargs['num_actions'] = self.envy.num_actions()
        kwargs['observation_width'] = self.observation_vector(self.envy.get_observation()).shape[-1]

        self.model = MOTorch(
            module_type=    module_type,
            name=           self.name,
            **kwargs)

        self._zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       self._tbwr) if self._tbwr else None

        self._rlog.info(f'*** DQNActor *** initialized')

    # returns QVs for a single observation
    def _get_QVs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observations=observation)['qvs'].detach().cpu().numpy()

    # single call with a batch of observations
    def get_QVs_batch(self, observations:np.ndarray) -> np.ndarray:
        return self._get_QVs(observation=observations)

    # not used by DQNActor
    def _upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:
        raise RLException('not implemented, should not be used since DQNActor only updates with a batch')

    # updates NN
    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        return self.model.backward(**training_data)

    # publishes
    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            training_data: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
    ) -> None:
        if self._tbwr:
            for k,v in metrics.items():
                if k != 'qvs':
                    self._tbwr.add(value=v, tag=f'actor/{k}', step=self._upd_step)


    def save(self):
        self.model.save()


    def load(self):
        self.model.load()


    def __str__(self) -> str:
        return str(self.model)