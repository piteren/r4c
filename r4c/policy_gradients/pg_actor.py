from abc import ABC
import numpy as np
from pypaq.lipytools.plots import two_dim_multi
from torchness.motorch import MOTorch, Module
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from torchness.comoneural.avg_probs import avg_mm_probs
from typing import Optional, Dict, Any, List

from r4c.helpers import NUM, zscore_norm, discounted_return, movavg_return
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
            discount: float=                        0.95,   # discount factor for discounted returns
            use_mavg: bool=                         True,   # use MovAvg (moving average, reversed) to calculate discounted returns
            mavg_factor: float=                     0.3,    # MovAvg factor
            do_zscore: bool=                        True,   # apply zscore norm to discounted returns
            **kwargs):

        TrainableActor.__init__(
            self,
            envy=   envy,
            name=   name,
            **kwargs)
        self.envy = envy  # to update type (for pycharm only)

        # some overrides and updates
        kwargs['num_actions'] = self.envy.num_actions()
        kwargs['observation_width'] = self.observation_vector(self.envy.get_observation()).shape[-1]

        self.model = MOTorch(
            module_type=    module_type,
            name=           self.name,
            **kwargs)

        self.discount = discount
        self.use_mavg = use_mavg
        self.movavg_factor = mavg_factor
        self.do_zscore = do_zscore

        self._zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       self._tbwr) if self._tbwr else None

        self._rlog.info(f'*** PGActor *** (NN based) initialized')
        self._rlog.info(f'> discount: {self.discount}')
        # TODO: add logs

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

    # splits rewards into episode rewards
    @staticmethod
    def _split_rewards(rewards, terminals) -> List[List[float]]:
        episode_rewards = []
        cep = []
        for r, t in zip(rewards, terminals):
            cep.append(r)
            if t:
                episode_rewards.append(cep)
                cep = []
        if cep: episode_rewards.append(cep)
        return episode_rewards

    # extracts from a batch + prepares dreturns
    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:

        episode_rewards = PGActor._split_rewards(batch['rewards'], batch['terminals'])

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

        if self.research_mode:
            # inspect each axis of observations
            oL = np.split(batch['observations'], batch['observations'].shape[-1], axis=-1)
            two_dim_multi(
                ys=     oL,
                names=  [f'obs_{ix}' for ix in range(len(oL))])

            # inspect rewards and all 4 types of dreturns
            dret_mavg = []
            dret_disc = []
            for rs in PGActor._split_rewards(batch['rewards'], batch['terminals']):
                dret_mavg += movavg_return(rewards=rs, factor=self.movavg_factor)
                dret_disc += discounted_return(rewards=rs, discount=self.discount)
            dret_mavg_norm = zscore_norm(dret_mavg)
            dret_disc_norm = zscore_norm(dret_disc)
            two_dim_multi(
                ys=     [
                    batch['rewards'],
                    dret_mavg,
                    dret_disc,
                    dret_mavg_norm,
                    dret_disc_norm],
                names=  [
                    'rewards',
                    'dret_mavg',
                    'dret_disc',
                    'dret_mavg_norm',
                    'dret_disc_norm'],
                legend_loc= 'lower left')


    def save(self):
        self.model.save()


    def load(self):
        self.model.load()


    def __str__(self):
        return self.model.__str__()