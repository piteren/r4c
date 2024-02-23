import numpy as np
from pypaq.pms.base import POINT
from pypaq.lipytools.pylogger import get_child
from torchness.motorch import MOTorch, Module
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from torchness.comoneural.avg_probs import avg_mm_probs
from typing import Optional, Dict, Any

from r4c.actor import ProbTRActor
from r4c.policy_gradients.pg_actor_module import PGActorModule



class PGActor(ProbTRActor):
    """ Policy Gradient Trainable Actor, MOTorch (NN) based """

    def __init__(
            self,
            model_type: type(MOTorch)=      MOTorch,
            module_type: type(Module)=      PGActorModule,
            motorch_point: Optional[POINT]= None,
            **kwargs):

        ProbTRActor.__init__(self, **kwargs)

        self.model = model_type(
            module_type=        module_type,
            name=               self.name,
            num_actions=        self.envy.num_actions,
            observation_width=  self.observation_width,
            seed=               self.seed,
            logger=             get_child(self.logger),
            hpmser_mode=        self.hpmser_mode,
            **(motorch_point or {}))

        self._zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       self.tbwr) if self.tbwr else None

    def _get_probs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observation=observation)['probs'].cpu().detach().numpy()

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        return self.model.backward(
            observation=    training_data['observation'],
            action=         training_data['action'],
            dreturn=        training_data['dreturn'])

    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
    ) -> None:

        if self.tbwr:

            self.tbwr.add_histogram(values=batch['observation'], tag='observation', step=self._upd_step)

            metrics.pop('logits')

            probs = metrics.pop('probs').cpu().detach().numpy()
            pm = avg_mm_probs(probs)
            for k in pm:
                self.tbwr.add(value=pm[k], tag=f'actor/{k}', step=self._upd_step)

            zeroes = metrics.pop('zeroes')
            self._zepro.process(zeroes=zeroes, step=self._upd_step)

            for k,v in metrics.items():
                self.tbwr.add(value=v, tag=f'actor/{k}', step=self._upd_step)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()

    def __str__(self) -> str:
        nfo =  f'{super().__str__()}\n'
        nfo += str(self.model)
        return nfo