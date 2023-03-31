import numpy as np
from torchness.motorch import Module, MOTorch
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import Optional, Dict, Any

from r4c.policy_gradients.actor_critic.ac_critic_module import ACCriticModule



class ACCritic:

    def __init__(
            self,
            observation_width: int,
            num_actions: int,
            name: str=                              'ACCritic',
            module_type: Optional[type(Module)]=    ACCriticModule,
            tbwr: Optional=                         None,
            **kwargs):

        self.model = MOTorch(
            module_type=        module_type,
            name=               name,
            observation_width=  observation_width,
            num_actions=        num_actions,
            **kwargs)

        self._tbwr = tbwr
        self._upd_step = 0

        self.zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tag_pfx=    'critic_nane',
            tbwr=       self._tbwr) if self._tbwr else None

    def get_qvs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observations=observation)['qvs'].detach().cpu().numpy()

    def update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        self._upd_step += 1
        return self.model.backward(
            observations=           training_data['observations'],
            actions_taken=          training_data['actions'],
            next_observations_qvs=  training_data['next_observations_qvs'],
            next_actions_probs=     training_data['next_actions_probs'],
            rewards=                training_data['rewards'])

    def publish(self, metrics: Dict[str,Any]):

        if self._tbwr:

            zeroes = metrics.pop('critic_zeroes')
            self.zepro.process(zs=zeroes, step=self._upd_step)

            metrics.pop('critic_qvs')
            for k, v in metrics.items():
                self._tbwr.add(value=v, tag=f'critic/{k[7:]}', step=self._upd_step)