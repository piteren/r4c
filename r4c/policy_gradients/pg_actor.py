from abc import ABC
import numpy as np
from pypaq.pms.base import POINT
from torchness.motorch import MOTorch, Module
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from torchness.comoneural.avg_probs import avg_mm_probs
from typing import Optional, Dict, Any

from r4c.helpers import NUM, zscore_norm, discounted_return, movavg_return, split_rewards, plot_obs_act, plot_rewards
from r4c.actor import TrainableActor
from r4c.envy import FiniteActionsRLEnvy
from r4c.policy_gradients.pg_actor_module import PGActorModule



# Policy Gradient Trainable Actor, NN based
class PGActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            module_type: Optional[type(Module)]=    PGActorModule,
            discount: float=                        0.95,   # discount factor for discounted returns
            use_mavg: bool=                         True,   # use MovAvg (moving average, reversed) to calculate discounted returns
            mavg_factor: float=                     0.3,    # MovAvg factor
            do_zscore: bool=                        True,   # apply zscore norm to discounted returns
            motorch_point: Optional[POINT]=         None,
            **kwargs):

        TrainableActor.__init__(self, envy=envy, **kwargs)
        self.envy = envy  # to update type (for pycharm only)

        self.discount = discount
        self.use_mavg = use_mavg
        self.movavg_factor = mavg_factor
        self.do_zscore = do_zscore

        motorch_point = motorch_point or {}
        motorch_point['num_actions'] = self.envy.num_actions()
        motorch_point['observation_width'] = self.observation_vector(self.envy.get_observation()).shape[-1]

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

    # prepares policy probs
    def get_policy_probs(self, observation:np.ndarray) -> np.ndarray:
        return self.model(observations=observation)['probs'].detach().cpu().numpy()

    # returns policy action based on policy probs
    def get_action(
            self,
            observation: np.ndarray,
            explore: bool = False,
            sample: bool=   False,
    ) -> NUM:

        if explore:
            return int(np.random.choice(self.envy.num_actions()))

        probs = self.get_policy_probs(observation)
        if sample: return np.random.choice(self.envy.num_actions(), p=probs)
        else:      return np.argmax(probs)

    # extracts from a batch + prepares dreturns
    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:

        episode_rewards = split_rewards(batch['rewards'], batch['terminals'])

        dreturns = []
        if self.use_mavg:
            for rs in episode_rewards:
                dreturns += movavg_return(rewards=rs, factor=self.movavg_factor)
        else:
            for rs in episode_rewards:
                dreturns += discounted_return(rewards=rs, discount=self.discount)
        if self.do_zscore:
            dreturns = zscore_norm(dreturns)

        return {
            'observations': batch['observations'],
            'actions':      batch['actions'],
            'dreturns':     np.asarray(dreturns)}

    # updates NN
    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        return self.model.backward(
            observations=   training_data['observations'],
            actions_taken=  training_data['actions'],
            dreturns=       training_data['dreturns'])


    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            training_data: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
            inspect: bool,
    ) -> None:

        if self._tbwr:

            probs = metrics.pop('probs').cpu().detach().numpy()
            pm = avg_mm_probs(probs)
            for k in pm:
                self._tbwr.add(value=pm[k], tag=f'actor/{k}', step=self._upd_step)

            zeroes = metrics.pop('zeroes')
            self._zepro.process(zs=zeroes, step=self._upd_step)

            metrics.pop('logits')
            for k,v in metrics.items():
                self._tbwr.add(value=v, tag=f'actor/{k}', step=self._upd_step)

        if inspect:
            plot_obs_act(observations=batch['observations'], actions=batch['actions'])
            plot_rewards(
                rewards=        batch['rewards'],
                terminals=      batch['terminals'],
                discount=       self.discount,
                movavg_factor=  self.movavg_factor)


    def save(self):
        self.model.save()


    def load(self):
        self.model.load()


    def __str__(self) -> str:
        nfo =  f'{super().__str__()}\n'
        nfo += f'> discount: {self.discount}\n'
        nfo += f'> use_mavg: {self.use_mavg}\n'
        nfo += f'> movavg_factor: {self.movavg_factor}\n'
        nfo += f'> do_zscore: {self.do_zscore}\n'
        nfo += str(self.model)
        return nfo