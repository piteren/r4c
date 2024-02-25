import numpy as np
from pypaq.pms.base import POINT
from torchness.motorch import Module
from typing import Optional, Dict, Any

from r4c.critic import FiniTRCritic, MOTRCritic
from r4c.policy_gradients.actor_critic.ac_critic_module import ACCriticModule


class ACCritic(FiniTRCritic, MOTRCritic):

    def __init__(self, module_type:Optional[type(Module)]=ACCriticModule, **kwargs):
        super().__init__(module_type=module_type, **kwargs)

    def _critic_motorch_point(self) -> POINT:
        p = super()._critic_motorch_point()
        p['num_actions'] = self.actor.envy.num_actions
        return p

    def get_value(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observation=observation)['qvs'].detach().cpu().numpy()

    def build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        return {k: batch[k] for k in ['observation','action','next_observation_qvs','next_action_probs','reward']}

    def update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        return self.model.backward(**training_data)

    def publish(self, metrics:Dict[str,Any]):
        if self.actor.tbwr:
            metrics.pop('critic_qvs')
            super().publish(metrics)