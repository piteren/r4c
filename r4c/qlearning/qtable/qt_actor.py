import numpy as np
from pypaq.lipytools.files import prep_folder, w_pickle, r_pickle
from typing import Dict, List, Any, Union

from r4c.qlearning.ql_actor import QLearningActor



class QTable:
    """ QTable stores QVs for observations.
    Here implemented as a Dict {observation_hash: QVs}, where:
    - observation hash is a string,
    - QVs are stored with numpy array. """

    def __init__(self, width:int):
        self.__width = width
        self.__table: Dict[str, np.ndarray] = {}  # {observation_hash: QVs}
        self.__keys: List[np.ndarray] = []

    @staticmethod
    def __hash(observation:np.ndarray) -> str:
        """ baseline hash of np.ndarray """
        return str(observation)

    def __init_hash(self, ha:str):
        self.__table[ha] = np.zeros(self.__width, dtype=float)

    def get_QVs(self, observation:np.ndarray) -> np.ndarray:
        ha = QTable.__hash(observation)
        if ha not in self.__table:
            self.__init_hash(ha)
            self.__keys.append(observation)
        return self.__table[ha]

    def put_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: np.ndarray,
    ):
        ha = QTable.__hash(observation)
        if ha not in self.__table:
            self.__init_hash(ha)
            self.__keys.append(observation)
        self.__table[ha][action] = new_qv

    def __str__(self):
        s = f'length: {len(self.__table) if self.__table else "<empty>"}\n'
        if self.__table:
            for k in sorted(self.__table.keys()):
                s += f'{k} : {self.__table[k]}\n'
        return s[:-1]


class QTableActor(QLearningActor):
    """ implements QLearningActor with QTable """

    def __init__(self, update_rate:float, **kwargs):
        super().__init__(**kwargs)
        self.update_rate = update_rate
        self.qtable = QTable(self.envy.num_actions)
        self.logger.info('*** QTableActor *** initialized')
        self.logger.info(f'> update_rate: {self.update_rate}')

    def _get_QVs(self, observation:np.ndarray) -> np.ndarray:
        """ returns QVs for given observation """
        return self.qtable.get_QVs(observation)

    def _upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: np.ndarray,
    ) -> float:
        """ updates QV and returns ~loss (TD Error) """
        old_qv = self._get_QVs(observation)[action]
        diff = new_qv - old_qv # TD Error
        self.qtable.put_QV(
            observation=    observation,
            action=         action,
            new_qv=         old_qv + self.update_rate * diff)
        return abs(float(diff))

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates QV """
        actor_metrics = super()._update(training_data)
        loss = 0.0
        for obs, act, nqv in zip(training_data['observation'], training_data['action'], training_data['new_qv']):
            loss += self._upd_QV(
                observation=    obs,
                action=         act,
                new_qv=         nqv)
        actor_metrics['loss'] = loss
        return actor_metrics

    def save(self):
        save_data = {
            'update_rate': self.update_rate,
            'qtable':      self.qtable}
        folder = self.save_dir
        prep_folder(folder)
        w_pickle(save_data, f'{folder}/qt.data')

    def load(self):
        saved_data = r_pickle(f'{self.save_dir}/qt.data')
        self.update_rate = saved_data['update_rate']
        self.qtable = saved_data['qtable']

    def __str__(self):
        nfo = f'{super().__str__()}\n'
        nfo += f'> update_rate: {self.update_rate}'
        return nfo