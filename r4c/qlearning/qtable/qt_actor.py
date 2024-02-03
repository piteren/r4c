import numpy as np
from pypaq.lipytools.files import prep_folder, w_pickle, r_pickle
from typing import Dict, List

from r4c.qlearning.ql_actor import QLearningActor



class QTable:
    """
    QTable stores QVs for observations.
    Here implemented as a Dict {observation_hash: QVs}, where:
    - observation hash is a string,
    - QVs are stored with numpy array.
    """

    def __init__(self, width:int):
        self.__width = width
        self.__table: Dict[str, np.ndarray] = {}  # {observation_hash: QVs}
        self.__keys: List[np.ndarray] = []

    # baseline hash of np.ndarray
    @staticmethod
    def __hash(observation:np.ndarray) -> str:
        return str(observation)

    def __init_hash(self, ha:str):
        self.__table[ha] = np.zeros(self.__width, dtype=float)

    def get_QVs(self, observation:np.ndarray) -> np.ndarray:
        ha = QTable.__hash(observation)
        if ha not in self.__table:
            self.__init_hash(ha)
            self.__keys.append(observation)
        return self.__table[ha]

    def put_QV(self,
            observation: np.ndarray,
            action: int,
            new_qv: float):
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


# implements QLearningActor with QTable
class QTableActor(QLearningActor):

    def __init__(
            self,
            name: str=          'QTableActor',
            update_rate: float= 0.5,
            **kwargs):
        QLearningActor.__init__(self, name=name, **kwargs)
        self.update_rate = update_rate
        self.__qtable = QTable(self.envy.num_actions())
        self._rlog.info('*** QTableActor *** initialized')
        self._rlog.info(f'> update_rate: {self.update_rate}')

    # returns QVs for given observation
    def _get_QVs(self, observation:np.ndarray) -> np.ndarray:
        return self.__qtable.get_QVs(observation)

    # updates QV and returns ~loss (TD Error)
    def _upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:

        old_qv = self._get_QVs(observation)[action]
        diff = new_qv - old_qv # TD Error
        self.__qtable.put_QV(
            observation=    observation,
            action=         action,
            new_qv=         old_qv + self.update_rate * diff)

        return abs(diff)

    def save(self):
        save_data = {
            'update_rate': self.update_rate,
            'qtable':      self.__qtable}
        folder = self.save_dir
        prep_folder(folder)
        w_pickle(save_data, f'{folder}/qt.data')

    def load(self):
        saved_data = r_pickle(f'{self.save_dir}/qt.data')
        self.update_rate = saved_data['update_rate']
        self.__qtable = saved_data['qtable']

    def __str__(self):
        return f'QTableActor, QTable:\n{self.__qtable.__str__()}'