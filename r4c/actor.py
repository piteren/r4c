from abc import abstractmethod, ABC
import numpy as np
from pypaq.pytypes import NUM
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.moving_average import MovAvg
import time
from torchness.tbwr import TBwr
from typing import Optional, Dict, Any, Tuple, List

from r4c.envy import RLEnvy
from r4c.helpers import R4Cexception, plot_obs_act, plot_rewards
from r4c.experience_memory import ExperienceMemory


class Actor(ABC):
    """ Actor (abstract) """

    def __init__(
            self,
            envy: RLEnvy,
            name: Optional[str]=    None,
            add_stamp: bool=        True,
            sample_PL: float=       0.0,        # sampling probability of playing
            save_topdir: str=       '_models',
            logger: Optional=       None,
            loglevel: int=          20,
    ):

        if name is None:
            name = f'{self.__class__.__name__}'
        if add_stamp: name += f'_{stamp()}'
        self.name = name

        self.save_topdir = save_topdir
        self.save_dir = f'{self.save_topdir}/{self.name}'

        if not logger:
            logger = get_pylogger(
                folder= self.save_dir,
                level=  loglevel)
        self._rlog = logger
        self._rlog.info(f'*** {self.__class__.__name__} (Actor) : {self.name} *** initializes..')

        self.envy = envy
        self._rlog.info(f'> Envy: {self.envy.__class__.__name__}')

        self.sample_PL = sample_PL

    def _observation_vector(self, observation:object) -> np.ndarray:
        """ prepares vector (np.ndarray) from observation, first tries to get from RLEnvy """
        try:
            return self.envy.observation_vector(observation)
        except R4Cexception:
            raise R4Cexception ('TrainableActor should implement get_observation_vec()')

    @abstractmethod
    def _get_action(self, observation:np.ndarray) -> NUM:
        """ returns Actor action from the Actor policy based on the observation """
        pass

    @abstractmethod
    def save(self): pass

    @abstractmethod
    def load(self): pass

    def __str__(self):
        nfo =  f'{self.__class__.__name__} (Actor) : {self.name}\n'
        nfo += f'> observation width: {self._observation_vector(self.envy.get_observation()).shape[-1]}'
        return nfo


class TrainableActor(Actor, ABC):
    """ Plays & Learns on Envy """

    def __init__(
            self,
            exploration: float=         0.0,        # exploration probability while building experience
            sample_TR: float=           0.0,        # sampling probability of training
            batch_size: int=            64,
            mem_batches: Optional[int]= None,       # ExperienceMemory max size (in number of batches), for None is unlimited
            sample_memory: bool=        False,      # sample batch from memory or get_all and reset
            publish_TB: bool=           True,
            hpmser_mode: bool=          False,
            seed: int=                  123,
            **kwargs):

        Actor.__init__(self, **kwargs)

        self._is_training = False  # Actor manages its trainable state, by default is false, only run_train() changes

        self.seed = seed

        mem_max_size = batch_size * mem_batches if mem_batches is not None else None
        self.memory = ExperienceMemory(
            max_size=   mem_max_size,
            seed=       self.seed)
        self._rlog.info(f'> initialized ExperienceMemory of max size {mem_max_size}')
        self._sample_memory = sample_memory

        self.exploration = exploration
        self.sample_TR = sample_TR
        self.batch_size = batch_size

        self.hpmser_mode = hpmser_mode
        # early override
        if self.hpmser_mode:
            publish_TB = False

        self._tbwr = TBwr(logdir=self.save_dir) if publish_TB else None
        self._upd_step = 0  # global update step

        np.random.seed(self.seed)

        self._rlog.debug(self)

    @abstractmethod
    def _get_random_action(self) -> NUM:
        """ returns random action """
        pass

    def _move(self) -> Tuple[
        np.ndarray, # observation
        NUM,        # action
        float,      # reward
        np.ndarray, # next observation
    ]:
        """ single  move of Actor on Envy (observation > action > reward) """

        # eventually (safety) reset Envy
        if self.envy.is_terminal():
            self.envy.reset()

        observation = self.envy.get_observation()
        observation_vector = self._observation_vector(observation)

        if self._is_training and np.random.rand() < self.exploration:
            action = self._get_random_action()
        else:
            action = self._get_action(observation=observation_vector)

        reward = self.envy.run(action)

        next_observation = self.envy.get_observation()
        next_observation_vector = self._observation_vector(next_observation)

        return observation_vector, action, reward, next_observation_vector

    def run_play(
            self,
            steps: Optional[int]=   None,
            reset: bool=            True,
            break_terminal: bool=   True,
            picture: bool=          False,
    ) -> Tuple[
        List[np.ndarray],   # observations
        List[NUM],          # actions
        List[float],        # rewards
        List[np.ndarray],   # next observations
        List[bool],         # terminals
        List[bool],         # wons
    ]:
        """ Actor plays some steps on Envy and returns data """

        if steps is None:
            steps = self.envy.max_steps

        if steps is None:
            raise R4Cexception('Actor cannot play on Envy where max_steps is None and given steps is None')

        if reset:
            self.envy.reset()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        wons = []

        while len(actions) < steps:

            observation, action, reward, next_observation = self._move()

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation)
            terminals.append(self.envy.is_terminal())
            wons.append(self.envy.has_won())

            if picture:
                self.envy.render()

            if terminals[-1] and break_terminal:
                break

        if picture:
            plot_obs_act(observations=observations, actions=actions)
            kw = {'rewards': rewards}
            if 'discount' in self.__dict__:
                kw.update({
                    'terminals':    terminals,
                    'discount':     self.__dict__['discount'],
                })
            plot_rewards(**kw)

        return observations, actions, rewards, next_observations, terminals, wons

    def run_train(
            self,
            num_batches: int,
            test_freq: Optional[int]=       None,
            test_episodes: Optional[int]=   None,
            test_max_steps: Optional[int]=  None,   # max number of episode steps while testing
            break_ntests: Optional[int]=    None,   # breaks training after all test episodes succeeded N times in a row
            picture: bool=                  False,  # plots and renders while testing
    ) -> dict:
        """ RL training procedure, returns dict with some training stats """

        self.envy.reset()
        self.memory.clear()

        loss_mavg = MovAvg()
        lossL = []

        n_won = 0                   # number of wins while training
        n_terminals = 0             # number of terminal states reached while training
        last_terminals = 0          # previous number of terminal states

        succeeded_row_curr = 0      # current number of succeeded tests in a row
        succeeded_row_max = 0       # max number of succeeded tests in a row

        batch_ix = 1                # index, starts from 1
        stime = time.time()

        self._is_training = True

        while batch_ix <= num_batches:

            ### play for a batch of data

            observations, actions, rewards, next_observations, terminals, wons = self.run_play(
                steps=          self.batch_size,
                reset=          False,
                break_terminal= False)

            n_won += sum(wons)
            n_terminals += sum(terminals)

            # store experience in memory
            self.memory.add(
                experience={
                    'observations':         observations,
                    'actions':              actions,
                    'rewards':              rewards,
                    'next_observations':    next_observations,
                    'terminals':            terminals,
                    'wons':                 wons})

            ### update

            batch = self.memory.get_sample(self.batch_size) if self._sample_memory else self.memory.get_all()
            training_data = self._build_training_data(batch=batch)

            metrics = self._update(training_data=training_data)
            lossL.append(loss_mavg.upd(metrics['loss']))

            self._publish(batch=batch, metrics=metrics)
            self._upd_step += 1

            ### test

            if test_freq and batch_ix % test_freq == 0:

                term_nfo = f'{n_terminals}(+{n_terminals - last_terminals})'
                loss_nfo = f'{loss_mavg():.4f}'
                tr_nfo = f'# {batch_ix:4} term:{term_nfo:11}: loss:{loss_nfo:7}'
                last_terminals = n_terminals

                self._is_training = False

                # single episode
                _, actions, rewards, _, _, wons = self.run_play(
                    steps=          test_max_steps,
                    reset=          True,
                    break_terminal= True,
                    picture=        picture)
                rewards_nfo = f'{sum(rewards):.1f}'
                ts_one_nfo = f'1TS: {len(actions):4} actions, return {rewards_nfo:5} ({" won" if sum(wons) else "lost"})'

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

                    self._rlog.info(f'{tr_nfo} -- {ts_one_nfo} -- {ts_nfo}')

                self._is_training = True

            batch_ix += 1

            if break_ntests is not None and succeeded_row_curr == break_ntests: break

        self._is_training = False
        self._rlog.info(f'### Training finished, time taken: {time.time() - stime:.2f}sec')

        return {
            'n_actions':            (batch_ix-1)*self.batch_size,   # total number of training actions
            'lossL':                lossL,
            'n_terminals':          n_terminals,
            'n_won':                n_won,
            'n_updates_done':       batch_ix-1,
            'succeeded_row_max':    succeeded_row_max}

    @abstractmethod
    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """ extracts data from a batch + eventually adds new """
        pass

    @abstractmethod
    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]:
        """ updates policy or value function (Actor, Critic ar any other component), returns metrics with 'loss' """
        pass

    def test_on_episodes(self, n_episodes:int=10, max_steps:Optional[int]=None) -> Tuple[float, float, float]:
        """ plays n episodes, returns tuple with won_factor & avg_reward """
        n_won = 0
        sum_rewards = 0
        sum_actions = 0
        for e in range(n_episodes):
            observations, _, rewards, _, _, wons = self.run_play(
                steps=          max_steps,
                reset=          True,
                break_terminal= True)
            n_won += sum(wons)
            sum_rewards += sum(rewards)
            sum_actions += len(observations)
        return n_won/n_episodes, sum_rewards/n_episodes, sum_actions/n_episodes

    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
    ) -> None:
        """ publishes to TB """
        if self._tbwr:
            self._tbwr.add(value=metrics['loss'], tag=f'actor/loss', step=self._upd_step)