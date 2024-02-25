import numpy as np
from pypaq.pytypes import NUM
from torchness.motorch import MOTorch, Module
from typing import Dict, Any

from r4c.critic import TrainableCritic
from r4c.policy_gradients.actor_critic.ac_actor import ACActor
from r4c.policy_gradients.ppo.ppo_actor_motorch import MOTorch_PPO
from r4c.policy_gradients.ppo.ppo_actor_module import PPOActorModule
from r4c.policy_gradients.ppo.ppo_critic import PPOCritic


class PPOActor(ACActor):

    ACTOR_TR_DATA_KEYS = ['observation', 'action', 'advantage', 'logprob']

    def __init__(
            self,
            model_type: type(MOTorch)=          MOTorch_PPO,
            module_type: type(Module)=          PPOActorModule,
            critic_type: type(TrainableCritic)= PPOCritic,
            **kwargs):
        super().__init__(
            model_type=     model_type,
            module_type=    module_type,
            critic_type=    critic_type,
            **kwargs)

    def get_action(self, observation:np.ndarray) -> Dict[str,NUM]:
        """ adds computation of value and logprob """
        ad = super().get_action(observation=observation)
        ad['value'] = self.critic.get_value(observation=observation)
        prob_action = ad['probs'][ad['action']]
        ad['logprob'] = np.log(prob_action)
        return ad

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ prepares training data for Actor and Critic """

        dk = ['observation', 'action']
        training_data = {k: batch[k] for k in dk}
        # INFO: PPO computes dreturn in a custom way, here is a classic baseline
        training_data['dreturn'] = self._prepare_dreturn(reward=batch['reward'], terminal=batch['terminal'])

        training_data['logprob'] = batch['logprob']
        training_data['advantage'] = training_data['dreturn'] - np.squeeze(batch['value'])

        return training_data