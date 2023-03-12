from abc import ABC
import numpy as np
from torchness.motorch import Module
from typing import Optional, Dict, Any

from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.actor_critic_shared.ac_shared_actor_module import ACSharedActorModule



class ACSharedActor(PGActor, ABC):

    def __init__(
            self,
            name: str=                              'ACSharedActor',
            module_type: Optional[type(Module)]=    ACSharedActorModule,
            **kwargs):
        PGActor.__init__(
            self,
            name=           name,
            module_type=    module_type,
            **kwargs)

    def update_with_experience(
            self,
            batch: Dict[str,np.ndarray],
            inspect: bool
    ) -> Dict[str, Any]:

        out = self.model.backward(
            observations=   batch['observations'],
            actions_taken=  batch['actions'],
            qv_labels=      batch['dreturns'])

        out.pop('qvs')
        out.pop('logits')
        out.pop('value')
        if 'probs' in out:
            out['probs'] = out['probs'].cpu().detach().numpy()

        return out