import numpy as np
import torch
from pypaq.pytypes import NUM
from typing import Dict, Any

from r4c.helpers import update_terminal_values
from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.ppo.ppo_actor_module import PPOActorModule
from r4c.policy_gradients.ppo.ppo_critic import PPOCritic


class PPOActor(PGActor):

    def __init__(self, **kwargs):

        # split kwargs assuming that Critic kwargs start with 'critic_'
        c_kwargs = {k[7:]: kwargs[k] for k in kwargs if k.startswith('critic_')}
        for k in c_kwargs:
            kwargs.pop(f'critic_{k}')

        PGActor.__init__(self, module_type=PPOActorModule, **kwargs)

        self.critic = PPOCritic(actor=self, **c_kwargs)

    def _get_action(self, observation:np.ndarray) -> Dict[str,NUM]:
        """ adds computation of value and logprob """
        ad = super()._get_action(observation=observation)
        ad['value'] = self.critic.get_value(observation=observation)
        prob_action = ad['probs'][ad['action']]
        ad['logprob'] = np.log(prob_action)
        return ad

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ prepares actor and critic data """

        training_data = super()._build_training_data(batch) # observation, action, dreturn
        # TODO: PPO in special way calculates dreturns (cleanrl ppo.py #214)
        for k in ['logprob','reward','value','terminal','next_observation']:
            training_data[k] = batch[k]

        next_value = self.critic.get_value(observation=batch['next_observation'])
        update_terminal_values(value=next_value, terminal=batch['terminal'])
        training_data['next_value'] = next_value

        for k in training_data:
            print(k, training_data[k].shape, training_data[k])

        """
        # get QV of action
        qvs = self.critic.get_qvs(batch['observation']) # QVs of current observation
        training_data['dreturn'] = qvs[range(len(batch['action'])), batch['action']] # get QV of selected action

        # get QVs of next observation, those come without gradients, which is ok - no target backpropagation
        next_observation_qvs = self.critic.get_qvs(batch['next_observation'])
        update_terminal_QVs(
            qvs=        next_observation_qvs,
            terminal=  batch['terminal'])
        training_data['next_observation_qvs'] = next_observation_qvs
        training_data['next_action_probs'] = self._get_probs(batch['next_observation'])  # get next_observation action_probs (with Actor policy)
        """
        return training_data

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates both NNs (actor + critic) """

        actor_training_data = {k: training_data[k] for k in ['observation','action','dreturn']}
        actor_metrics = super()._update(training_data=actor_training_data)

        critic_training_data = {k: training_data[k] for k in ['observation','action','next_observation_qvs','next_action_probs','reward']}
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