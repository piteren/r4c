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
            module_type: type(Module)=      PGActorModule,
            motorch_point: Optional[POINT]= None,
            **kwargs):

        ProbTRActor.__init__(self, **kwargs)

        motorch_point = motorch_point or {}
        motorch_point['num_actions'] = self.envy.num_actions()
        motorch_point['observation_width'] = self._observation_vector(self.envy.get_observation()).shape[-1]

        self.model = MOTorch(
            module_type=    module_type,
            name=           self.name,
            seed=           self.seed,
            logger=         get_child(self._rlog),
            hpmser_mode=    self.hpmser_mode,
            **motorch_point)

        self._zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       self._tbwr) if self._tbwr else None

    def _get_policy_probs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observations=observation)['probs'].detach().cpu().numpy()

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        return self.model.backward(
            observations=   training_data['observations'],
            actions_taken=  training_data['actions'],
            dreturns=       training_data['dreturns'])

    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
    ) -> None:

        if self._tbwr:

            self._tbwr.add_histogram(values=batch['observations'], tag='observations', step=self._upd_step)

            metrics.pop('logits')

            probs = metrics.pop('probs').cpu().detach().numpy()
            pm = avg_mm_probs(probs)
            for k in pm:
                self._tbwr.add(value=pm[k], tag=f'actor/{k}', step=self._upd_step)

            zeroes = metrics.pop('zeroes')
            self._zepro.process(zeroes=zeroes, step=self._upd_step)

            for k,v in metrics.items():
                self._tbwr.add(value=v, tag=f'actor/{k}', step=self._upd_step)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()

    def __str__(self) -> str:
        nfo =  f'{super().__str__()}\n'
        nfo += str(self.model)
        return nfo