import numpy as np
from pypaq.pytypes import NUM
from pypaq.pms.base import POINT
from torchness.motorch import MOTorch, Module
from typing import Dict

from r4c.envy import CASRLEnvy
from r4c.actor import MOTRActor
from r4c.critic import TrainableCritic
from r4c.policy_gradients.actor_critic.ac_actor import AbstrACActor
from r4c.policy_gradients.ppo.ppo_actor_motorch import MOTorch_PPO
from r4c.policy_gradients.ppo.ppo_critic import PPOCritic
from r4c.policy_gradients.ppo_cas.ppo_cas_actor_module import PPOCASActorModule


class PPOCASActor(AbstrACActor, MOTRActor):
    """ PPO for continuous action space (CAS) MOTRActor with Critic,
    this implementation is very similar to PPOActor,
    but PPOActor cannot be used since it supports finite action space """

    ACTOR_TR_DATA_KEYS = ['observation', 'action', 'advantage', 'logprob']

    def __init__(
            self,
            envy: CASRLEnvy,
            model_type: type(MOTorch)=          MOTorch_PPO,
            module_type: type(Module)=          PPOCASActorModule,
            critic_type: type(TrainableCritic)= PPOCritic,
            **kwargs
    ):
        super().__init__(
            envy=           envy,
            model_type=     model_type,
            module_type=    module_type,
            critic_type=    critic_type,
            **kwargs)
        self.envy = envy

    def _actor_motorch_point(self) -> POINT:
        p = super()._actor_motorch_point()
        p['action_width'] = self.envy.action_width
        return p

    def get_action(self, observation:np.ndarray) -> Dict[str,NUM]:
        """ adds computation of value and logprob (does not return probs) """
        fwdd = self.model(observation=observation)
        return {
            'action':   fwdd['action'].cpu().detach().numpy(),
            'logprob':  fwdd['logprob'].cpu().detach().numpy(),
            'value':    self.critic.get_value(observation=observation)}

    def _build_training_data(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """ prepares training data for Actor and Critic,
        implementation copied from PPOActor """

        dk = ['observation', 'action']
        training_data = {k: batch[k] for k in dk}
        # INFO: PPO computes dreturn in a custom way, here is a classic baseline
        training_data['dreturn'] = self._prepare_dreturn(reward=batch['reward'], terminal=batch['terminal'])

        training_data['logprob'] = batch['logprob']
        training_data['advantage'] = training_data['dreturn'] - np.squeeze(batch['value'])

        return training_data