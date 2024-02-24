from abc import abstractmethod, ABC
import numpy as np
from pypaq.pytypes import NUM
from pypaq.pms.base import POINT
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.moving_average import MovAvg
import time
from torchness.tbwr import TBwr
from torchness.comoneural.avg_probs import avg_mm_probs
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from torchness.motorch import MOTorch, Module
from typing import Optional, Dict, Any, Tuple

from r4c.envy import RLEnvy, FiniteActionsRLEnvy
from r4c.helpers import R4Cexception, zscore_norm, da_return, split_reward, plot_obs_act, plot_reward
from r4c.experience_memory import ExperienceMemory


class Actor(ABC):

    def __init__(
            self,
            envy: RLEnvy,
            name: Optional[str]=    None,
            add_stamp: bool=        True,
            save_topdir: str=       '_models',
            logger: Optional=       None,
            loglevel: int=          20,
    ):

        if name is None:
            name = f'{self.__class__.__name__}'
        self.add_stamp = add_stamp
        if self.add_stamp:
            name += f'_{stamp()}'
        self.name = name

        self.save_topdir = save_topdir
        self.save_dir = f'{self.save_topdir}/{self.name}'

        if not logger:
            logger = get_pylogger(
                folder= self.save_dir,
                level=  loglevel)
        self.logger = logger
        self.logger.info(f'*** {self.__class__.__name__} (Actor) : {self.name} *** initializes..')

        self.envy = envy

    def _observation_vector(self, observation:object) -> np.ndarray:
        """ prepares vector (np.ndarray) from observation, first tries to get from RLEnvy """
        try:
            return self.envy.observation_vector(observation)
        except R4Cexception:
            raise R4Cexception ('TrainableActor should implement get_observation_vec()')

    @abstractmethod
    def get_action(self, observation:np.ndarray) -> Dict[str,NUM]:
        """ Actor (agent) gets observation and executes an action with the policy,
        action is based on observation,
        returns a dict with some data like probs, action, logprob, value.. """
        pass

    @abstractmethod
    def save(self): pass

    @abstractmethod
    def load(self): pass

    @property
    def observation_width(self) -> int:
        return self._observation_vector(self.envy.get_observation()).shape[-1]

    def __str__(self):
        nfo =  f'{self.__class__.__name__} (Actor) : {self.name}\n'
        nfo += f'> observation width: {self.observation_width}'
        return nfo


class TrainableActor(Actor, ABC):
    """ TrainableActor is an Actor that Plays & Learns on Envy """

    def __init__(
            self,
            exploration: float,                 # exploration probability while building experience (TR)
            discount: float,                    # discount factor for discounted returns
            do_zscore: bool,                    # apply zscore norm to discounted returns
            batch_size: int,
            mem_batches: Optional[int]= None,   # ExperienceMemory max size (in number of batches), for None is unlimited
            sample_memory: bool=        False,  # sample batch of single samples from memory or get_all and reset memory
            publish_TB: bool=           True,
            hpmser_mode: bool=          False,
            seed: int=                  123,
            **kwargs):

        super().__init__(**kwargs)

        """ Actor manages its trainable state, by default is false, only run_train() changes.
        Trainable state may be used for example by _get_action for exploration. """
        self._is_training = False

        self.seed = seed

        mem_max_size = batch_size * mem_batches if mem_batches is not None else None
        self.memory = ExperienceMemory(
            max_size=   mem_max_size,
            seed=       self.seed,
            logger=     get_child(self.logger))
        self._sample_memory = sample_memory

        self.discount = discount
        self.do_zscore = do_zscore
        self.exploration = exploration
        self.batch_size = batch_size
        self.upd_step = 0  # global update step

        self.hpmser_mode = hpmser_mode
        # early override
        if self.hpmser_mode:
            publish_TB = False

        self.tbwr = TBwr(logdir=self.save_dir) if publish_TB else None

        np.random.seed(self.seed)

    @abstractmethod
    def _get_random_action(self) -> NUM:
        """ returns 100% random action """
        pass

    def _move(self) -> Dict[str,NUM]:
        """ executes single move of Actor on Envy """

        # eventually (safety) reset Envy in case it reached terminal state and has not been reset by the user
        if self.envy.is_terminal():
            self.envy.reset()

        observation = self.envy.get_observation()
        observation_vector = self._observation_vector(observation)

        ad = self.get_action(observation=observation_vector)
        if self._is_training and np.random.rand() < self.exploration:
            ad['action'] = self._get_random_action()

        reward = self.envy.run(ad['action'])

        next_observation = self.envy.get_observation()
        next_observation_vector = self._observation_vector(next_observation)

        md = {
            'observation':      observation_vector,
            'reward':           reward,
            'next_observation': next_observation_vector}
        md.update(ad)

        return md

    def run_play(
            self,
            steps: Optional[int]=   None,
            break_terminal: bool=   True,
            picture: bool=          False,
    ) -> Dict[str,np.ndarray]:
        """ Actor plays some steps on Envy and returns data
        implementation below is a baseline and returns:
        {   < any keys & data returned by _move() > +
            'terminal':            List[bool],
            'won'                  List[bool]} """

        if steps is None:
            steps = self.envy.max_steps

        if steps is None:
            raise R4Cexception('Actor cannot play on Envy where max_steps is None and given steps is None')

        ed = {k: [] for k in ['action','terminal','won']}
        while len(ed['action']) < steps:

            md = self._move()
            for k in md:
                if k not in ed:
                    ed[k] = []
                ed[k].append(md[k])

            ed['terminal'].append(self.envy.is_terminal())
            ed['won'].append(self.envy.has_won())

            if picture:
                self.envy.render()

            if ed['terminal'][-1] and break_terminal:
                break

        ed = {k: np.asarray(ed[k]) for k in ed}

        if picture:
            plot_obs_act(observation=ed['observation'], action=ed['action'])
            kw = {'reward': ed['reward']}
            if 'discount' in self.__dict__:
                kw.update({
                    'terminal': ed['terminal'],
                    'discount': self.__dict__['discount'],
                })
            plot_reward(**kw)

        return ed

    def run_train(
            self,
            num_batches: int,
            test_freq: Optional[int]=       None,
            test_episodes: Optional[int]=   None,
            test_max_steps: Optional[int]=  None,   # max number of episode steps while testing
            break_ntests: Optional[int]=    None,   # breaks training after all test episodes succeeded N times in a row
            picture: bool=                  False,  # plots and renders while testing
    ) -> dict:
        """ generic RL training procedure, returns dict with some training stats """

        self.envy.reset()
        self.memory.reset()

        loss_mavg = MovAvg()
        lossL = []

        n_won = 0               # number of wins while training
        n_terminal = 0          # number of terminal states reached while training
        last_terminal = 0       # previous number of terminal states

        succeeded_row_curr = 0  # current number of succeeded tests in a row
        succeeded_row_max = 0   # max number of succeeded tests in a row

        batch_ix = 1            # index, starts from 1
        stime = time.time()

        self._is_training = True

        while batch_ix <= num_batches:

            ### play for a batch of data

            ed = self.run_play(
                steps=          self.batch_size,
                break_terminal= False)

            n_won += sum(ed['won'])
            n_terminal += sum(ed['terminal'])

            self.memory.add(experience=ed)

            ### update

            batch = self.memory.get_sample(self.batch_size) if self._sample_memory else self.memory.get_all()
            training_data = self._build_training_data(batch=batch)

            metrics = self._update(training_data=training_data)
            lossL.append(loss_mavg.upd(metrics['loss']))

            self._publish(metrics)
            self.upd_step += 1

            ### test

            if test_freq and batch_ix % test_freq == 0:

                term_nfo = f'{n_terminal}(+{n_terminal - last_terminal})'
                loss_nfo = f'{loss_mavg():.4f}'
                tr_nfo = f'# {batch_ix:4} term:{term_nfo:11}: loss:{loss_nfo:7}'
                last_terminal = n_terminal

                self._is_training = False

                # single episode
                self.envy.reset()
                ed = self.run_play(
                    steps=          test_max_steps,
                    break_terminal= True,
                    picture=        picture)
                reward_nfo = f'{sum(ed["reward"]):.1f}'
                ts_one_nfo = f'1TS: {len(ed["action"]):4} actions, return {reward_nfo:5} ({" won" if sum(ed["won"]) else "lost"})'

                # few tests
                if test_episodes:
                    avg_won, avg_return, avg_actions = self.test_on_episodes(
                        n_episodes= test_episodes,
                        max_steps=  test_max_steps)
                    ts_nfo = f'{test_episodes}xTS: avg_won:{avg_won * 100:.1f}%, avg_return:{avg_return:.2f}, avg_actions:{int(avg_actions)}'

                    if avg_won == 1:
                        succeeded_row_curr += 1
                        if succeeded_row_curr > succeeded_row_max: succeeded_row_max = succeeded_row_curr
                    else:
                        succeeded_row_curr = 0

                    self.logger.info(f'{tr_nfo} -- {ts_one_nfo} -- {ts_nfo}')

                self._is_training = True

            batch_ix += 1

            if break_ntests is not None and succeeded_row_curr == break_ntests:
                break

        self._is_training = False
        self.logger.info(f'### Training finished, time taken: {time.time() - stime:.2f}sec')

        return {
            'n_action':             (batch_ix-1)*self.batch_size,   # total number of training actions
            'lossL':                lossL,
            'n_terminal':           n_terminal,
            'n_won':                n_won,
            'n_updates_done':       batch_ix-1,
            'succeeded_row_max':    succeeded_row_max}

    # TODO: add typing
    def _get_dreturn(self, reward:np.ndarray, terminal:np.ndarray) -> np.ndarray:
        episode_reward = split_reward(reward, terminal)
        dreturn = []
        for rs in episode_reward:
            dreturn.append(da_return(reward=rs, discount=self.discount))
        dreturn = np.concatenate(dreturn, axis=-1)
        if self.do_zscore:
            dreturn = zscore_norm(dreturn)
        return dreturn

    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ extracts observation + action from a batch + prepares dreturn """
        dk = ['observation', 'action']
        training_data = {k: batch[k] for k in dk}
        training_data['dreturn'] = self._get_dreturn(reward=batch['reward'], terminal=batch['terminal'])
        return training_data

    @abstractmethod
    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates policy or value function (Actor, Critic ar any other component), returns metrics with 'loss' """
        return {'observation': training_data['observation']}

    def test_on_episodes(self, n_episodes:int=10, max_steps:Optional[int]=None) -> Tuple[float, float, float]:
        """ plays n episodes, returns tuple with won_factor & avg_reward """
        n_won = 0
        sum_reward = 0
        sum_action = 0
        for e in range(n_episodes):
            self.envy.reset()
            ed = self.run_play(
                steps=          max_steps,
                break_terminal= True)
            n_won += sum(ed['won'])
            sum_reward += sum(ed['reward'])
            sum_action += len(ed['observation'])
        return n_won/n_episodes, sum_reward/n_episodes, sum_action/n_episodes

    def _publish(self, metrics:Dict[str,Any]) -> None:

        if self.tbwr:

            self.tbwr.add_histogram(
                values= metrics.pop('observation'),
                tag=    'observation',
                step=   self.upd_step)

            for k,v in metrics.items():
                self.tbwr.add(value=v, tag=f'actor/{k}', step=self.upd_step)

    def __str__(self) -> str:
        nfo =  f'{super().__str__()}\n'
        nfo += f'> discount:  {self.discount}\n'
        nfo += f'> do_zscore: {self.do_zscore}\n'
        return nfo


class FiniTRActor(TrainableActor, ABC):
    """ FiniTRActor is a TrainableActor for FiniteActionsRLEnvy """

    def __init__(self, envy:FiniteActionsRLEnvy, **kwargs):
        super().__init__(envy=envy, **kwargs)
        self.envy = envy  # to update the type (for pycharm)

    def _get_random_action(self) -> NUM:
        return int(np.random.choice(self.envy.num_actions))


class ProbTRActor(FiniTRActor, ABC):
    """ ProbTRActor is a Probabilistic FiniTRActor, implements:
    - sample_PL - sampling probability while playing
    - sample_TR - sampling probability while training """

    def __init__(self, sample_PL:float, sample_TR:float, **kwargs):
        super().__init__(**kwargs)
        self.sample_PL = sample_PL
        self.sample_TR = sample_TR

    @abstractmethod
    def _get_probs(self, observation:np.ndarray) -> np.ndarray:
        """ runs agent policy to return policy probs """
        pass

    def get_action(self, observation:np.ndarray) -> Dict[str,NUM]:
        probs = self._get_probs(observation)
        sample =        (self._is_training and np.random.rand() < self.sample_TR
                  or not self._is_training and np.random.rand() < self.sample_PL)
        action = np.random.choice(self.envy.num_actions, p=probs) if sample else np.argmax(probs)
        return {'probs':probs, 'action':action}

    def _publish(self, metrics:Dict[str,Any]) -> None:

        if self.tbwr:

            probs = metrics.pop('probs').cpu().detach().numpy()
            pm = avg_mm_probs(probs)
            for k in pm:
                self.tbwr.add(value=pm[k], tag=f'actor/{k}', step=self.upd_step)

            super()._publish(metrics)


class MOTRActor(TrainableActor, ABC):
    """ MOTRActor is a MOTorch (NN model) based TrainableActor """

    def __init__(
            self,
            module_type: Optional[type(Module)],
            model_type: type(MOTorch)=      MOTorch,
            motorch_point: Optional[POINT]= None,
            **kwargs):

        super().__init__(**kwargs)

        self.model = model_type(
            module_type=        module_type,
            **self._actor_motorch_point(),
            **(motorch_point or {}))

        self._zepro = ZeroesProcessor(
            intervals=  (10, 50, 100),
            tbwr=       self.tbwr) if self.tbwr else None

    def _actor_motorch_point(self) -> POINT:
        """ prepares Actor specific motorch_point addon """
        return {
            'name':                 self.name,
            'observation_width':    self.observation_width,
            'seed':                 self.seed,
            'logger':               get_child(self.logger),
            'hpmser_mode':          self.hpmser_mode}

    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ update with NN backprop """
        actor_metrics = self.model.backward(**training_data)
        actor_metrics['observation'] = training_data['observation']
        return actor_metrics

    def _publish(self, metrics:Dict[str,Any]) -> None:
        if self.tbwr:
            self._zepro.process(zeroes=metrics.pop('zeroes'), step=self.upd_step)
            super()._publish(metrics)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()

    def __str__(self) -> str:
        nfo = f'{super().__str__()}\n'
        nfo += str(self.model)
        return nfo