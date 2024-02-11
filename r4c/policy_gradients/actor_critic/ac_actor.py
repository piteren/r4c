import numpy as np
from typing import Dict, Any

from r4c.helpers import update_terminal_QVs
from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.actor_critic.ac_critic import ACCritic


class ACActor(PGActor):

    def __init__(
            self,
            name: str=                      'ACActor',
            critic_class: type(ACCritic)=   ACCritic,
            **kwargs):

        # split kwargs assuming that Critic kwargs start with 'critic_'
        c_kwargs = {k[7:]: kwargs[k] for k in kwargs if k.startswith('critic_')}
        for k in c_kwargs:
            kwargs.pop(f'critic_{k}')
        for k in ['logger','loglevel']:
            if k in kwargs:
                c_kwargs[k] = kwargs[k]

        PGActor.__init__(self, name=name, **kwargs)

        self.critic = critic_class(
            observation_width=  self._observation_vector(self.envy.get_observation()).shape[-1],
            num_actions=        self.envy.num_actions(),
            tbwr=               self._tbwr,
            hpmser_mode=        self.hpmser_mode,
            seed=               kwargs['seed'],
            **c_kwargs)

        self._rlog.info('*** ACActor *** initialized')
        self._rlog.info(f'> critic: {critic_class.__name__}')

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ prepares actor and critic data """

        training_data = {
            'observation': batch['observation'],
            'action':      batch['action'],
            'reward':      batch['reward']}

        # get QV of action
        qvs = self.critic.get_qvs(batch['observation']) # QVs of current observation
        training_data['dreturn'] = qvs[range(len(batch['action'])), batch['action']] # get QV of selected action

        # get QVs of next observation, those come without gradients, which is ok - no target backpropagation
        next_observation_qvs = self.critic.get_qvs(batch['next_observation'])
        update_terminal_QVs(
            qvs=        next_observation_qvs,
            terminal=   batch['terminal'])
        training_data['next_observation_qvs'] = next_observation_qvs
        training_data['next_action_probs'] = self._get_probs(batch['next_observation'])  # get next_observation action_probs (with Actor policy)

        return training_data

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates both NNs """

        actor_training_data = {k: training_data[k] for k in ['observation','action','dreturn']}
        actor_metrics = super()._update(training_data=actor_training_data)

        critic_training_data = {k: training_data[k] for k in ['observation','actions','next_observation_qvs','next_actions_probs','reward']}
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