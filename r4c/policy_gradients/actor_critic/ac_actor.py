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
        for k in c_kwargs: kwargs.pop(f'critic_{k}')

        PGActor.__init__(
            self,
            name=           name,
            **kwargs)

        self.critic = critic_class(
            observation_width=  self._observation_vector(self.envy.get_observation()).shape[-1],
            num_actions=        self.envy.num_actions(),
            tbwr=               self._tbwr,
            seed=               kwargs['seed'],
            **c_kwargs)

        self._rlog.info('*** ACActor *** initialized')
        self._rlog.info(f'> critic: {critic_class.__name__}')

    # prepares actor and critic data
    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:

        training_data = {
            'observations': batch['observations'],
            'actions':      batch['actions'],
            'rewards':      batch['rewards']}

        # get QV of action
        qvs = self.critic.get_qvs(batch['observations']) # QVs of current observations
        training_data['dreturns'] = qvs[range(len(batch['actions'])), batch['actions']] # get QV of selected actions

        # get QVs of next observations, those come without gradients, which is ok - no target backpropagation
        next_observations_qvs = self.critic.get_qvs(batch['next_observations'])
        update_terminal_QVs(
            qvs=        next_observations_qvs,
            terminals=  batch['terminals'])
        training_data['next_observations_qvs'] = next_observations_qvs
        training_data['next_actions_probs'] = self._get_policy_probs(batch['next_observations'])  # get next_observations actions_probs (with Actor policy)

        return training_data

    # updates both NNs
    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:

        actor_training_data = {k: training_data[k] for k in ['observations','actions','dreturns']}
        actor_metrics = super()._update(training_data=actor_training_data)

        critic_training_data = {k: training_data[k] for k in ['observations','actions','next_observations_qvs','next_actions_probs','rewards']}
        critic_metrics = self.critic.update(training_data=critic_training_data)

        # merge metrics
        for k in critic_metrics:
            actor_metrics[f'critic_{k}'] = critic_metrics[k]

        return actor_metrics

    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            training_data: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
            inspect: bool,
    ) -> None:

        critic_metrics = {k: metrics[k] for k in metrics if k.startswith('critic')}
        for k in critic_metrics: metrics.pop(k)
        super()._publish(batch=batch, training_data=training_data, metrics=metrics, inspect=inspect)
        self.critic.publish(critic_metrics)

        if inspect:
            """
            print(f'\nBatch size: {len(batch)}')
            print(f'observations: {observations.shape}, {observations[0]}')
            print(f'actions: {actions.shape}, {actions[0]}')
            print(f'rewards {rewards.shape}, {rewards[0]}')
            print(f'next_observations {next_observations.shape}, {next_observations[0]}')
            print(f'terminals {terminals.shape}, {terminals[0]}')
            print(f'next_actions_probs {next_actions_probs.shape}, {next_actions_probs[0]}')
            print(f'qvss {qvss.shape}, {qvss[0]}')
            print(f'qv_actions {qv_actions.shape}, {qv_actions[0]}')
            print(f'actions_OH {actions_OH.shape}, {actions_OH[0]}')
            print(f'next_observations_qvs {next_observations_qvs.shape}, {next_observations_qvs[0]}')
            """
            pass