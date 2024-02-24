from abc import ABC
import numpy as np
from pypaq.pms.base import POINT
from pypaq.lipytools.pylogger import get_child
from torchness.motorch import MOTorch, Module
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import Optional, Dict, Any

from r4c.qlearning.ql_actor import QLearningActor
from r4c.qlearning.dqn.dqn_actor_module import DQNModel


class DQNActor(QLearningActor, ABC):
    """ DQN (NN based) QLearningActor """

    def __init__(
            self,
            module_type: Optional[type(Module)]=    DQNModel,
            motorch_point: Optional[POINT]=         None,
            **kwargs):

        super().__init__(**kwargs)

        self.model = MOTorch(
            module_type=        module_type,
            name=               self.name,
            num_actions=        self.envy.num_actions,
            observation_width=  self.observation_width,
            seed=               self.seed,
            logger=             get_child(self.logger),
            hpmser_mode=        self.hpmser_mode,
            **(motorch_point or {}))

        self._zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       self.tbwr) if self.tbwr else None

    def _get_QVs(self, observation:np.ndarray) -> np.ndarray:
        """ returns QVs for a single observation """
        return self.model(observation=observation)['qvs'].detach().cpu().numpy()

    def get_QVs_batch(self, observation:np.ndarray) -> np.ndarray:
        """ single call with a batch of observation """
        return self._get_QVs(observation=observation)

    def _upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:
        """ not used since DQNActor updates with a batch only """
        raise NotImplementedError

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        actor_metrics = self.model.backward(**training_data)
        actor_metrics['observation'] = training_data['observation']
        return actor_metrics

    def _publish(self, metrics:Dict[str,Any]) -> None:
        if self.tbwr:
            if 'qvs' in metrics:
                metrics.pop('qvs')
            self._zepro.process(zeroes=metrics.pop('zeroes'), step=self.upd_step)
            super()._publish(metrics)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()

    def __str__(self) -> str:
        nfo = f'{super().__str__()}\n'
        nfo += str(self.model)
        return nfo