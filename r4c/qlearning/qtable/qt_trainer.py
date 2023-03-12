from r4c.qlearning.ql_trainer import QLearningTrainer
from r4c.qlearning.qtable.qt_actor import QTableActor


# Trainer for QTableActor, sets Actor update_rate
class QTableTrainer(QLearningTrainer):

    def __init__(
            self,
            actor: QTableActor,
            update_rate: float,
            **kwargs):

        QLearningTrainer.__init__(self, actor=actor, **kwargs)
        self.actor = actor
        self.actor.set_update_rate(update_rate)

        self._rlog.info('*** QTableTrainer *** initialized')
        self._rlog.info(f'> actor update_rate: {update_rate}')