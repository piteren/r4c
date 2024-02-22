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

        FiniTRCritic.__init__(self, **kwargs)

        self.model = MOTorch(
            module_type=        module_type,
            name=               self.name,
            observation_width=  self.actor.observation_width,
            num_actions=        self.actor.envy.num_actions,
            discount=           self.actor.discount,
            seed=               self.actor.seed,
            logger=             get_child(self.logger),
            **(motorch_point or {}))

        self._upd_step = 0

        self.zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tag_pfx=    'critic_nane',
            tbwr=       self.actor.tbwr) if self.actor.tbwr else None

    def get_qvs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observation=observation)['qvs'].detach().cpu().numpy()

    def update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        self._upd_step += 1
        return self.model.backward(
            observation=            training_data['observation'],
            action_taken=           training_data['action'],
            next_observation_qvs=   training_data['next_observation_qvs'],
            next_action_probs=      training_data['next_action_probs'],
            reward=                 training_data['reward'])

    def publish(self, metrics:Dict[str,Any]):

        if self.actor.tbwr:

            zeroes = metrics.pop('critic_zeroes')
            self.zepro.process(zeroes=zeroes, step=self._upd_step)

            metrics.pop('critic_qvs')
            for k, v in metrics.items():
                self.actor.tbwr.add(value=v, tag=f'critic/{k[7:]}', step=self._upd_step)