from abc import ABC
import numpy as np
from torchness.motorch import MOTorch, Module
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
            seed: int=                              123,
            **kwargs):

        QLearningActor.__init__(
            self,
            name=   name,
            seed=   seed,
            **kwargs)

        # some overrides and updates
        if 'logger' in kwargs: kwargs.pop('logger')     # NNWrap will always create own logger (since then it is not given) with optionally given level
        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self.observation_vector(self._envy.get_observation()).shape[-1]

        self.model = MOTorch(
            module_type=    module_type,
            name=           self.name,
            seed=           seed,
            **kwargs)

        self._rlog.info(f'*** DQNActor : {self.name} *** initialized')


    def _get_QVs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observations=observation)['logits'].detach().cpu().numpy()

    # single call with a batch of observations (..no diff with _get_QVs)
    def get_QVs_batch(self, observations:np.ndarray) -> np.ndarray:
        return self.model(observations=observations)['logits'].detach().cpu().numpy()


    def _upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:
        raise RLException('not implemented, should not be used since DQN_Actor updates only with batches')

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            batch: Dict[str,np.ndarray],
            inspect: bool,
    ) -> Dict[str, Any]:

        full_qvs = np.zeros_like(batch['observations'])
        mask = np.zeros_like(batch['observations'])
        for v,pos in zip(batch['new_qvs'], enumerate(batch['actions'])):
            full_qvs[pos] = v
            mask[pos] = 1
        batch['full_qvs'] = full_qvs
        batch['mask'] = mask

        for k in ['observations', 'actions', 'new_qvs', 'full_qvs', 'mask']:
            self._rlog.log(5, f'>> {k}, shape: {batch[k].shape}\n{batch[k]}')

        out = self.model.backward(
            observations=   batch['observations'],
            labels=         batch['full_qvs'],
            mask=           batch['mask'])

        out.pop('logits')

        return out

    def _get_save_topdir(self) -> str:
        return self.model['save_topdir']

    def save(self):
        self.model.save()

    def __str__(self) -> str:
        return str(self.model)