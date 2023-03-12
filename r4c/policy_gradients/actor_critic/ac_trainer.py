import numpy as np

from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.actor_critic.ac_critic import ACCritic
from r4c.policy_gradients.pg_trainer import PGTrainer


# RL Trainer for Actor & Critic acting on RLEnvy, implements Actor & Critic update
class ACTrainer(PGTrainer):

    def __init__(
            self,
            actor: PGActor,
            critic_class: type(ACCritic),
            **kwargs):

        # split kwargs assuming that Critic kwargs starts with 'critic_'
        c_kwargs = {k[7:]: kwargs[k] for k in kwargs if k.startswith('critic_')}
        for k in c_kwargs: kwargs.pop(f'critic_{k}')

        PGTrainer.__init__(
            self,
            actor=  actor,
            **kwargs)
        self.actor = actor # INFO: type "upgrade" for pycharm editor

        self.critic = critic_class(
            envy=   kwargs['envy'],
            seed=   kwargs['seed'],
            **c_kwargs)

        self._rlog.info('*** ACTrainer *** initialized')
        self._rlog.info(f'> critic: {critic_class.__name__}')

    # converts one dim arr of ints into two dim one-hot array
    def _actions_OH_encoding(self, actions:np.ndarray) -> np.ndarray:
        hot = np.zeros((len(actions), self.envy.num_actions()))
        hot[np.arange(len(actions)), actions] = 1
        return hot

    # sets terminal states QVs to zeroes
    def _update_terminal_QVs(self, qvs, terminals):
        zeroes = np.zeros(self.envy.num_actions())
        for i in range(len(terminals)):
            if terminals[i]:
                qvs[i] = zeroes
        return qvs

    # update is performed for both: Actor & Critic
    def _update_actor(self, inspect=False) -> dict:

        batch = self.memory.get_all()

        next_actions_probs = self.actor.get_policy_probs(batch['next_observations']) # get next_observations actions_probs (with Actor policy)

        # get QVs of current observations
        qvss = self.critic.get_qvs_batch(batch['observations'])
        qv_actions = qvss[np.arange(batch['actions'].shape[-1]), batch['actions']] # get QV of selected actions
        batch['dreturns'] = qv_actions

        # update Actor
        metrics = self.actor.update_with_experience(
            batch=      batch,
            inspect=    inspect)

        actions_OH = self._actions_OH_encoding(batch['actions'])

        # get QVs of next observations
        next_actions_qvs = self.critic.get_qvs_batch(batch['next_observations'])
        next_actions_qvs = self._update_terminal_QVs(next_actions_qvs, terminals=batch['terminals'])

        batch['actions_OH'] = actions_OH
        batch['next_actions_qvs'] = next_actions_qvs
        batch['next_actions_probs'] = next_actions_probs

        # TODO: replace prints with logger
        """
        if inspect:
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
            print(f'next_action_qvs {next_actions_qvs.shape}, {next_actions_qvs[0]}')
        """

        # update Critic
        crt_metrics = self.critic.update_with_experience(
            batch=      batch,
            inspect=    inspect)

        metrics['zeroes'] += crt_metrics.pop('zeroes')

        for k in crt_metrics:
            metrics[f'critic_{k}'] = crt_metrics[k]

        return metrics