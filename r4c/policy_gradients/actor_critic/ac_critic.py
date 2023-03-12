import numpy as np
from torchness.motorch import Module
from typing import Optional, Dict, Any

from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.actor_critic.ac_critic_module import ACCriticModule
from r4c.helpers import RLException


class ACCritic(PGActor):

    def __init__(
            self,
            name: str=                              'ACCritic',
            module_type: Optional[type(Module)]=    ACCriticModule,
            **kwargs):
        PGActor.__init__(
            self,
            name=           name,
            module_type=    module_type,
            **kwargs)

    # Critic does not have a policy
    def get_policy_probs(self, observation:np.ndarray) -> np.ndarray:
        raise RLException('not implemented since should not be called')


    def get_qvs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observations=observation)['qvs'].detach().cpu().numpy()


    def update_with_experience(
            self,
            batch: Dict[str,np.ndarray],
            inspect: bool
    ) -> Dict[str, Any]:

        out = self.model.backward(
            observations=           batch['observations'],
            actions_taken_OH=       batch['actions_OH'],
            next_observations_qvs=  batch['next_observations_qvs'],
            next_actions_probs=     batch['next_actions_probs'],
            rewards=                batch['rewards'])

        out.pop('qvs')

        return out