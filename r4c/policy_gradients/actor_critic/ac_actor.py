from abc import abstractmethod, ABC
import numpy as np
from typing import Dict, Any

from r4c.helpers import update_terminal_values
from r4c.actor import TrainableActor
from r4c.critic import TrainableCritic
from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.actor_critic.ac_critic import ACCritic


class AbstrACActor(TrainableActor, ABC):
    """ AbstrACActor is an abstract of TrainableActor with TrainableCritic (AC) """

    # since _build_training_data() prepares data for both Actor and Critic,
    # keys below specify which part of the data is for an Actor
    # this info is used by _update(), which updates also both Actor and Critic
    ACTOR_TR_DATA_KEYS = ['observation', 'action', 'dreturn']

    def __init__(self, critic_type:type(TrainableCritic), **kwargs):

        # split kwargs assuming that Critic kwargs start with 'critic_'
        c_kwargs = {k[7:]: kwargs[k] for k in kwargs if k.startswith('critic_')}
        for k in c_kwargs:
            kwargs.pop(f'critic_{k}')

        super().__init__(**kwargs)

        self.critic = critic_type(actor=self, **c_kwargs)

    @abstractmethod
    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ prepares Actor and Critic data """
        pass

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates both Actor and Critic """

        actor_training_data = {k: training_data[k] for k in self.ACTOR_TR_DATA_KEYS}
        actor_metrics = super()._update(actor_training_data)

        critic_training_data = self.critic.build_training_data(batch=training_data)
        critic_metrics = self.critic.update(critic_training_data)

        # merge metrics
        for k in critic_metrics:
            actor_metrics[f'critic_{k}'] = critic_metrics[k]

        return actor_metrics

    def _publish(self, metrics:Dict[str,Any]) -> None:
        critic_metrics = {k: metrics[k] for k in metrics if k.startswith('critic')}
        for k in critic_metrics:
            metrics.pop(k)
        super()._publish(metrics)
        self.critic.publish(critic_metrics)

    def save(self):
        super().save()
        self.critic.save()

    def load(self):
        super().load()
        self.critic.load()


class ACActor(PGActor, AbstrACActor):

    def __init__(self, critic_type:type(TrainableCritic)=ACCritic, **kwargs):
        super().__init__(critic_type=critic_type, **kwargs)

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ prepares Actor and Critic data """

        dk = ['observation','action','reward']
        training_data = {k: batch[k] for k in dk}

        # get QV of action
        qvs = self.critic.get_value(batch['observation']) # QVs of current observation
        training_data['dreturn'] = qvs[range(len(batch['action'])), batch['action']] # get QV of selected action

        # get QVs of next observation, those come without gradients, which is ok - no target backpropagation
        next_observation_qvs = self.critic.get_value(batch['next_observation'])
        update_terminal_values(
            value=      next_observation_qvs,
            terminal=   batch['terminal'])
        training_data['next_observation_qvs'] = next_observation_qvs
        training_data['next_action_probs'] = self._get_probs(batch['next_observation'])  # get next_observation action_probs (with Actor policy)

        return training_data