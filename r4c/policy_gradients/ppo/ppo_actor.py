import numpy as np
from pypaq.pytypes import NUM
from torchness.motorch import MOTorch, Module
from typing import Dict, Any

from r4c.helpers import update_terminal_values
from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.ppo.ppo_actor_motorch import MOTorch_PPO
from r4c.policy_gradients.ppo.ppo_actor_module import PPOActorModule
from r4c.policy_gradients.ppo.ppo_critic import PPOCritic


class PPOActor(PGActor):
    """ PPO Actor, MOTorch (NN) based """

    def __init__(
            self,
            model_type: type(MOTorch)=      MOTorch_PPO,
            module_type: type(Module)=      PPOActorModule,
            **kwargs):

        # split kwargs assuming that Critic kwargs start with 'critic_'
        c_kwargs = {k[7:]: kwargs[k] for k in kwargs if k.startswith('critic_')}
        for k in c_kwargs:
            kwargs.pop(f'critic_{k}')

        PGActor.__init__(self, model_type=model_type, module_type=module_type, **kwargs)

        self.critic = PPOCritic(actor=self, **c_kwargs)

    def _get_action(self, observation:np.ndarray) -> Dict[str,NUM]:
        """ adds computation of value and logprob """
        ad = super()._get_action(observation=observation)
        ad['value'] = self.critic.get_value(observation=observation)
        prob_action = ad['probs'][ad['action']]
        ad['logprob'] = np.log(prob_action)
        return ad

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ prepares Actor and Critic training data """

        # TODO: PPO computes dreturn in custom way (cleanrl ppo.py #214)
        training_data = super()._build_training_data(batch) # observation, action, dreturn

        training_data['logprob'] = batch['logprob']
        training_data['advantage'] = training_data['dreturn'] - np.squeeze(batch['value'])

        return training_data

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates both NNs (actor + critic) """

        actor_training_data = {k: training_data[k] for k in ['observation','action','advantage','logprob']}
        actor_metrics = super()._update(actor_training_data)

        critic_training_data = {k: training_data[k] for k in ['observation','dreturn']}
        critic_metrics = self.critic.update(training_data=critic_training_data)

        # merge metrics
        for k in critic_metrics:
            actor_metrics[f'critic_{k}'] = critic_metrics[k]

        return actor_metrics

    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
    ) -> None:
        critic_metrics = {k: metrics[k] for k in metrics if k.startswith('critic')}
        for k in critic_metrics:
            metrics.pop(k)
        super()._publish(batch=batch, metrics=metrics)
        self.critic.publish(critic_metrics)