import numpy as np
from pypaq.lipytools.plots import two_dim_multi

from r4c.helpers import zscore_norm, discounted_return, movavg_return
from r4c.policy_gradients.pg_actor import PGActor
from r4c.trainer import FATrainer


# Policy Gradient Trainer
class PGTrainer(FATrainer):

    def __init__(
            self,
            actor: PGActor,
            discount: float,    # discount factor for discounted returns
            use_mavg: bool,     # use MovAvg (moving average, reversed) to calculate discounted returns
            mavg_factor: float, # MovAvg factor
            do_zscore: bool,    # apply zscore norm to discounted returns
            **kwargs):

        FATrainer.__init__(self, actor=actor, **kwargs)
        self.actor = actor # INFO: just type "upgrade" for pycharm editor
        self.discount = discount
        self.use_mavg = use_mavg
        self.movavg_factor = mavg_factor
        self.do_zscore = do_zscore

        self._rlog.info('*** PGTrainer *** initialized')
        self._rlog.info(f'> discount: {self.discount}')

    # PGActor update method
    def _update_actor(self, inspect:bool=False) -> dict:

        batch = self.memory.get_all()

        ### prepare dreturns

        # split rewards into episodes
        episode_rewards = []
        cep = []
        for r,t in zip(batch['rewards'], batch['terminals']):
            cep.append(r)
            if t:
                episode_rewards.append(cep)
                cep = []
        if cep: episode_rewards.append(cep)

        dreturns = []
        if self.use_mavg:
            for rs in episode_rewards:
                dreturns += movavg_return(rewards=rs, factor=self.movavg_factor)
        else:
            for rs in episode_rewards:
                dreturns += discounted_return(rewards=rs, discount=self.discount)
        if self.do_zscore:
            dreturns = zscore_norm(dreturns)
        batch['dreturns'] = np.asarray(dreturns)

        if inspect:

            # inspect each axis of observations
            oL = np.split(batch['observations'], batch['observations'].shape[-1], axis=-1)
            two_dim_multi(
                ys=     oL,
                names=  [f'obs_{ix}' for ix in range(len(oL))])

            # inspect rewards and all 4 types of dreturns
            dret_mavg = []
            dret_disc = []
            for rs in episode_rewards:
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

        return self.actor.update_with_experience(
            batch=      batch,
            inspect=    inspect)