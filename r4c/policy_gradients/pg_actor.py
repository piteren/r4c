import numpy as np
from pypaq.pms.base import POINT
from torchness.motorch import Module
from typing import Dict, Any

from r4c.actor import ProbTRActor, MOTRActor
from r4c.policy_gradients.pg_actor_module import PGActorModule


class PGActor(ProbTRActor, MOTRActor):
    """ Policy Gradient MOTRActor """

    def __init__(self, module_type:type(Module)=PGActorModule, **kwargs):
        super().__init__(module_type=module_type, **kwargs)

    def _actor_motorch_point(self) -> POINT:
        p = super()._actor_motorch_point()
        p['num_actions'] = self.envy.num_actions
        return p

    def _get_probs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observation=observation)['probs'].cpu().detach().numpy()

    def _publish(self, metrics:Dict[str,Any]) -> None:
        if self.tbwr:
            if 'logits' in metrics:
                metrics.pop('logits')
            super()._publish(metrics)