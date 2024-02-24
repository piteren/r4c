import numpy as np
from pypaq.pms.base import POINT
from torchness.motorch import MOTorch, Module
from typing import Optional

from r4c.actor import ProbTRActor, MOTRActor
from r4c.policy_gradients.pg_actor_module import PGActorModule



class PGActor(ProbTRActor, MOTRActor):
    """ Policy Gradient MOTRActor """

    def __init__(
            self,
            model_type: type(MOTorch)=      MOTorch,
            module_type: type(Module)=      PGActorModule,
            motorch_point: Optional[POINT]= None,
            **kwargs):
        super().__init__(
            model_type=     model_type,
            module_type=    module_type,
            motorch_point=  motorch_point,
            **kwargs)

    def _actor_motorch_point(self) -> POINT:
        p = super()._actor_motorch_point()
        p['num_actions'] = self.envy.num_actions
        return p

    def _get_probs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observation=observation)['probs'].cpu().detach().numpy()