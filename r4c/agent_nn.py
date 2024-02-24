from abc import abstractmethod, ABC
import numpy as np
from pypaq.pms.base import POINT
from torchness.motorch import MOTorch, Module
from torchness.tbwr import TBwr
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import Optional, Dict, Any


class MOToAgent(ABC):
    """ MOToAgent is a MOTorch (NN model) based Agent for RL tasks """

    def __init__(
            self,
            tag: str,                               # actor / critic / ..
            module_type: Optional[type(Module)],
            model_type: type(MOTorch)=      MOTorch,
            motorch_point: Optional[POINT]= None,
            tbwr: Optional[TBwr]=           None,   # given or not by Agent
    ):

        self.model = model_type(
            module_type=    module_type,
            **self._agent_motorch_point(),
            **(motorch_point or {}))

        self.tag = tag
        self.tbwr = tbwr

        self.upd_step = 0
        self.zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tag_pfx=    f'{self.tag}_nane',
            tbwr=       self.tbwr) if self.tbwr else None

    @abstractmethod
    def _agent_motorch_point(self) -> POINT:
        """ prepares and allows to inject while init Agent specific motorch_point """
        pass

    def update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates Agent with NN backprop """
        metrics = self.model.backward(**training_data)
        self.upd_step += 1
        if 'logits' in metrics:
            metrics.pop('logits')
        return metrics

    def publish(self, metrics:Dict[str,Any]) -> None:
        if self.tbwr:
            self.zepro.process(zeroes=metrics.pop('zeroes'), step=self.upd_step)
            for k, v in metrics.items():
                self.tbwr.add(value=v, tag=f'{self.tag}/{k}', step=self.upd_step)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()

    def __str__(self) -> str:
        return f'MOToAgent build with: {self.model}'