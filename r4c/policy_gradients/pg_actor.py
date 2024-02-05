import numpy as np
from pypaq.pytypes import NUM
from pypaq.pms.base import POINT
from torchness.motorch import MOTorch, Module
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from torchness.comoneural.avg_probs import avg_mm_probs
from typing import Optional, Dict, Any

from r4c.helpers import zscore_norm, da_returns, split_rewards
from r4c.actor import TrainableActor
from r4c.envy import FiniteActionsRLEnvy
from r4c.policy_gradients.pg_actor_module import PGActorModule



class PGActor(TrainableActor):
    """ Policy Gradient Trainable Actor, MOTorch (NN) based """

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            module_type: Optional[type(Module)]=    PGActorModule,
            discount: float=                        0.95,   # discount factor for discounted returns
            do_zscore: bool=                        True,   # apply zscore norm to discounted returns
            motorch_point: Optional[POINT]=         None,
            **kwargs):

        TrainableActor.__init__(self, envy=envy, **kwargs)
        self.envy = envy  # to update the type (for pycharm)

        self.discount = discount
        self.do_zscore = do_zscore

        motorch_point = motorch_point or {}
        motorch_point['num_actions'] = self.envy.num_actions()
        motorch_point['observation_width'] = self._observation_vector(self.envy.get_observation()).shape[-1]

        self.model = MOTorch(
            module_type=    module_type,
            name=           self.name,
            seed=           self.seed,
            logger=         self._rlog,
            hpmser_mode=    self.hpmser_mode,
            **motorch_point)

        self._zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       self._tbwr) if self._tbwr else None

    def _get_policy_probs(self, observation:np.ndarray) -> np.ndarray:
        """ prepares policy probs """
        return self.model(observations=observation)['probs'].detach().cpu().numpy()

    def _get_random_action(self) -> NUM:
        return int(np.random.choice(self.envy.num_actions()))

    def _get_action(self, observation:np.ndarray) -> NUM:
        sample =        (self._is_training and np.random.rand() < self.sample_TR
                  or not self._is_training and np.random.rand() < self.sample_PL)
        probs = self._get_policy_probs(observation)
        if sample:
            return int(np.random.choice(self.envy.num_actions(), p=probs))
        else:
            return int(np.argmax(probs))

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ extracts from a batch + prepares dreturns """

        episode_rewards = split_rewards(batch['rewards'], batch['terminals'])

        dreturns = []
        for rs in episode_rewards:
            dreturns += da_returns(rewards=rs, discount=self.discount)
        if self.do_zscore:
            dreturns = zscore_norm(dreturns)

        return {
            'observations': batch['observations'],
            'actions':      batch['actions'],
            'dreturns':     np.asarray(dreturns)}

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
        nfo += f'> discount:  {self.discount}\n'
        nfo += f'> do_zscore: {self.do_zscore}\n'
        nfo += str(self.model)
        return nfo