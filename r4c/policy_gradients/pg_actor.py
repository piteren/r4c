from abc import ABC
import numpy as np
from torchness.motorch import MOTorch, Module
from typing import Optional, Dict, Any

from r4c.actor import TrainableActor
from r4c.envy import FiniteActionsRLEnvy
from r4c.policy_gradients.pg_actor_module import PGActorModule



# Policy Gradient Trainable Actor, NN based
class PGActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            name: str=                              'PGActor',
            module_type: Optional[type(Module)]=    PGActorModule,
            seed: int=                              123,
            **kwargs):

        TrainableActor.__init__(
            self,
            envy=   envy,
            name=   name,
            **kwargs)
        self._envy = envy  # to update type (for pycharm only)

        np.random.seed(seed)

        # some overrides and updates
        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self.observation_vector(self._envy.get_observation()).shape[-1]

        self.model = MOTorch(
            module_type=    module_type,
            name=           self.name,
            seed=           seed,
            **kwargs)

        self._rlog.info(f'*** PGActor : {self.name} *** (NN based) initialized')

    # prepares policy probs
    def get_policy_probs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observations=observation)['probs'].detach().cpu().numpy()

    # returns policy action based on policy probs
    def get_policy_action(self, observation:np.ndarray, sampled=False) -> int:
        probs = self.get_policy_probs(observation)
        if sampled: action = np.random.choice(self._envy.num_actions(), p=probs)
        else:       action = np.argmax(probs)
        return int(action)

    # updates self NN with batch of data
    def update_with_experience(
            self,
            batch: Dict[str,np.ndarray],
            inspect: bool,
    ) -> Dict[str, Any]:

        out = self.model.backward(
            observations=   batch['observations'],
            actions_taken=  batch['actions'],
            dreturns=       batch['dreturns'])

        out.pop('logits')
        if 'probs' in out:
            out['probs'] = out['probs'].cpu().detach().numpy()

        return out


    def _get_save_topdir(self) -> str:
        return self.model['save_topdir']


    def save(self):
        self.model.save()


    def __str__(self):
        return self.model.__str__()