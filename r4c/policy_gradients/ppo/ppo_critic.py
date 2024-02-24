import numpy as np
from torchness.motorch import Module
from typing import Optional, Dict

from r4c.critic import MOTRCritic
from r4c.policy_gradients.ppo.ppo_critic_module import PPOCriticModule


class PPOCritic(MOTRCritic):

    def __init__(self, module_type:Optional[type(Module)]=PPOCriticModule, **kwargs):
        super().__init__(module_type=module_type, **kwargs)

    def build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        return {k: batch[k] for k in ['observation','dreturn']}