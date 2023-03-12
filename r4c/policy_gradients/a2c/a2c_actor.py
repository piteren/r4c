import numpy as np
from pypaq.lipytools.plots import two_dim_multi
from torchness.motorch import Module
from typing import Optional, Dict, Any

from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.a2c.a2c_actor_module import A2CModule



class A2CActor(PGActor):

    def __init__(
            self,
            name: str=                              'A2CActor',
            module_type: Optional[type(Module)]=    A2CModule,
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

        actor_metrics = super().update_with_experience(batch,inspect)

        if inspect:
            ins_vals = {
                'dreturns':     batch['dreturns'],
                'value':        actor_metrics['value'].detach().cpu().numpy(),
                'advantage':    actor_metrics['advantage'].detach().cpu().numpy()}
            two_dim_multi(
                ys=         list(ins_vals.values()),
                names=      list(ins_vals.keys()),
                legend_loc= 'lower left')

        actor_metrics.pop('value')
        actor_metrics.pop('advantage')

        return actor_metrics