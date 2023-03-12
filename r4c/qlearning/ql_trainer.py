import numpy as np

from r4c.helpers import update_terminal_QVs
from r4c.trainer import FATrainer
from r4c.qlearning.ql_actor import QLearningActor


# Q-Learning Trainer for QLearningActor acting on FiniteActionsRLEnvy
class QLearningTrainer(FATrainer):

    def __init__(
            self,
            actor: QLearningActor,
            gamma: float,       # QLearning gamma (discount)
            **kwargs):

        FATrainer.__init__(self, actor=actor, **kwargs)
        self.actor = actor  # INFO: just type "upgrade" for pycharm editor
        self.gamma = gamma

        self._rlog.info(f'*** QLearningTrainer *** initialized')
        self._rlog.info(f'> gamma: {self.gamma}')


    def _update_actor(self, inspect=False) -> dict:

        batch = self.memory.get_sample(self.batch_size)

        next_observations_qvs = self.actor.get_QVs_batch(batch['next_observations'])

        update_terminal_QVs(
            qvs=        next_observations_qvs,
            terminals=  batch['terminals'])

        new_qvs = [r + self.gamma * max(no_qvs) for r, no_qvs in zip(batch['rewards'], next_observations_qvs)]
        batch['new_qvs'] = np.asarray(new_qvs)

        return self.actor.update_with_experience(
            batch=      batch,
            inspect=    inspect)