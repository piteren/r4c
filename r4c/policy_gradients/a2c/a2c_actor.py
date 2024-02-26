from torchness.motorch import Module
from typing import Optional, Dict, Any

from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.a2c.a2c_actor_module import A2CModule



class A2CActor(PGActor):

    def __init__(self, module_type:Optional[type(Module)]=A2CModule, **kwargs):
        super().__init__(module_type=module_type, **kwargs)

    def _publish(self, metrics:Dict[str,Any]) -> None:
        metrics.pop('value')
        metrics.pop('advantage')
        super()._publish(metrics)
