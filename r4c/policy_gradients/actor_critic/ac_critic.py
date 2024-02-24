import numpy as np
from pypaq.pms.base import POINT
from pypaq.lipytools.pylogger import get_child
from torchness.motorch import Module, MOTorch
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import Optional, Dict, Any

from r4c.critic import FiniTRCritic
from r4c.policy_gradients.actor_critic.ac_critic_module import ACCriticModule


class ACCritic(FiniTRCritic):

    def __init__(
            self,
            module_type: Optional[type(Module)]=    ACCriticModule,
            motorch_point: Optional[POINT]=         None,
            **kwargs):

        super().__init__(**kwargs)

        self.model = MOTorch(
            module_type=        module_type,
            name=               self.name,
            observation_width=  self.actor.observation_width,
            num_actions=        self.actor.envy.num_actions,
            discount=           self.actor.discount,
            seed=               self.actor.seed,
            logger=             get_child(self.logger),
            **(motorch_point or {}))

        self.zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tag_pfx=    'critic_nane',
            tbwr=       self.actor.tbwr) if self.actor.tbwr else None

    def get_value(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observation=observation)['qvs'].detach().cpu().numpy()

    def build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        return {k: batch[k] for k in ['observation','action','next_observation_qvs','next_action_probs','reward']}

    def update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        return self.model.backward(**training_data)

    def publish(self, metrics:Dict[str,Any]):

        if self.actor.tbwr:

            metrics.pop('critic_qvs')

            zeroes = metrics.pop('critic_zeroes')
            self.zepro.process(zeroes=zeroes, step=self.actor.upd_step)

            for k, v in metrics.items():
                self.actor.tbwr.add(value=v, tag=f'critic/{k[7:]}', step=self.actor.upd_step)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()

    def __str__(self) -> str:
        nfo =  f'{super().__str__()}\n'
        nfo += str(self.model)
        return nfo