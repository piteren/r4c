import numpy as np
from pypaq.pms.base import POINT
from torchness.motorch import MOTorch, Module
from typing import Optional, Dict, Any

from r4c.actor import MOTRActor
from r4c.qlearning.ql_actor import QLearningActor
from r4c.qlearning.dqn.dqn_actor_module import DQNModule


class DQNActor(QLearningActor, MOTRActor):
    """ DQN (NN based) QLearningActor """

    def __init__(
            self,
            model_type: type(MOTorch)=      MOTorch,
            module_type: type(Module)=      DQNModule,
            motorch_point: Optional[POINT]= None,
            **kwargs):
        super().__init__(
            model_type=     model_type,
            module_type=    module_type,
            motorch_point=  motorch_point,
            **kwargs)

    def _actor_motorch_point(self) -> POINT:
        p = super()._actor_motorch_point()
        p['num_actions'] = self.envy.num_actions
        return p

    def _observation_vector(self, observation:object) -> np.ndarray:
        """ since number of observations may be finite (in QTable is) in QLearning,
        observation_vector returned by super may be of int dtype,
        here it is converted to float for NN input """
        return super()._observation_vector(observation).astype(float)

    def _get_QVs(self, observation:np.ndarray) -> np.ndarray:
        """ returns QVs for a single observation """
        return self.model(observation=observation)['qvs'].detach().cpu().numpy()

    def get_QVs_batch(self, observation:np.ndarray) -> np.ndarray:
        """ single call with a batch of observation """
        return self._get_QVs(observation=observation)

    def _publish(self, metrics:Dict[str,Any]) -> None:
        if self.tbwr:
            metrics.pop('qvs')
            super()._publish(metrics)