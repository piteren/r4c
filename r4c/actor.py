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
from r4c.helpers import RLException, ExperienceMemory, plot_obs_act, plot_rewards


# just abstract Actor
class Actor(ABC):

    # returns Actor action based on observation according to Actor policy
    @abstractmethod
    def _get_action(self, observation:object) -> object: pass


# Plays & Learns on Envy
class TrainableActor(Actor, ABC):

    def __init__(
            self,
            envy: RLEnvy,
            name: Optional[str]=        None,
            add_stamp: bool=            True,
            exploration: float=         0.0,            # exploration probability while building experience
            sampled_TR: float=          0.0,            # sampling probability while building experience
            sampled_PL: float=          0.0,            # sampling probability while playing
            batch_size: int=            64,
            mem_batches: Optional[int]= None,           # ExperienceMemory max size (in number of batches)
            sample_memory: bool=        False,          # sample batch from memory or get_all and reset
            save_topdir: str=           '_models',
            logger: Optional=           None,
            loglevel: int=              20,
            publish_TB: bool=           True,
            hpmser_mode: bool=          False,
            seed: int=                  123):

        self.envy = envy

        if name is None:
            name = f'TrainableActor_{self.envy.__class__.__name__}'
        if add_stamp: name += f'_{stamp()}'
        self.name = name

        self.save_topdir = save_topdir

        self._rlog = logger or get_pylogger(
            folder= self.get_save_dir(),
            level=  loglevel)
        self._rlog.info(f'*** {self.__class__.__name__} (TrainableActor) : {self.name} *** initializes..')

        self.seed = seed

        mem_max_size = batch_size * mem_batches if mem_batches is not None else None
        self.memory = ExperienceMemory(
            max_size=   mem_max_size,
            seed=       self.seed)
        self._rlog.info(f'> initialized ExperienceMemory of max size {mem_max_size}')
        self._sample_memory = sample_memory

        self.exploration = exploration
        self.sampled_TR = sampled_TR
        self.sampled_PL = sampled_PL
        self.batch_size = batch_size

        self.hpmser_mode = hpmser_mode
        # early override
        if self.hpmser_mode:
            publish_TB = False

        self._tbwr = TBwr(logdir=self.get_save_dir()) if publish_TB else None
        self._upd_step = 0  # global update step

        np.random.seed(self.seed)

        self._rlog.debug(self)

    # prepares vector (np.ndarray) from observation, first tries to get from RLEnvy
    def _observation_vector(self, observation:object) -> np.ndarray:
        try:
            return self.envy.observation_vector(observation)
        except RLException:
            raise RLException ('TrainableActor should implement get_observation_vec()')

    # adds exploration & sampling
    @abstractmethod
    def _get_action(
            self,
            observation: np.ndarray,    # ..vector type
            explore: bool=  False,      # returns exploring action
            sample: bool=   False,      # samples action (from policy probability)
    ) -> NUM: pass

    # single  move of Actor on Envy (observation > action > reward)
    def _move(self, training:bool) -> Tuple[
        np.ndarray, # observation
        NUM,        # action
        float,      # reward
        np.ndarray, # next observation
    ]:

        # eventually (safety) reset Envy
        if self.envy.is_terminal():
            self.envy.reset()

        observation = self.envy.get_observation()
        observation_vector = self._observation_vector(observation)

        action = self._get_action(
            observation=    observation_vector,
            explore=        training and np.random.rand() < self.exploration,
            sample=         np.random.rand() < (self.sampled_TR if training else self.sampled_PL))

        reward = self.envy.run(action)

        next_observation = self.envy.get_observation()
        next_observation_vector = self._observation_vector(next_observation)

        return observation_vector, action, reward, next_observation_vector

    # plays some steps on Envy
    def run_play(
            self,
            steps: Optional[int]=   None,
            training: bool=         False,
            reset: bool=            True,
            break_terminal: bool=   True,
            inspect: bool=          False,
    ) -> Tuple[
        List[np.ndarray],   # observations
        List[NUM],          # actions
        List[float],        # rewards
        List[np.ndarray],   # next observations
        List[bool],         # terminals
        List[bool],         # wons
    ]:

        if steps is None:
            steps = self.envy.get_max_steps()

        if steps is None:
            raise RLException('Actor cannot play on Envy where max_steps is None and given steps is None')

        if reset: self.envy.reset()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        wons = []

        while len(actions) < steps:

            observation, action, reward, next_observation = self._move(training=training)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation)
            terminals.append(self.envy.is_terminal())
            wons.append(self.envy.won())

            if inspect: self.envy.render()

            if terminals[-1] and break_terminal:
                break

        if inspect:
            plot_obs_act(observations=observations, actions=actions)
            kw = {'rewards': rewards}
            # TODO: do it with method def _inspect(self, observations, actions, rewards, terminals) to override?
            if 'discount' in self.__dict__:
                kw.update({
                    'terminals':        terminals,
                    'discount':         self.__dict__['discount'],
                    'movavg_factor':    self.__dict__['movavg_factor'],
                })
            plot_rewards(**kw)

        return observations, actions, rewards, next_observations, terminals, wons

    # RL training procedure, returns dict with some training stats
    def run_train(
            self,
            num_batches: int,
            test_freq: Optional[int],
            test_episodes: Optional[int],
            test_max_steps: Optional[int] = None,   # max number of episode steps while testing
            inspect: bool=                  False,  # inspects data while updating / testing
            break_ntests: Optional[int]=    None,   # breaks training after all test episodes succeeded N times in a row
    ) -> dict:

        self.envy.reset()
        self.memory.clear()

        loss_mavg = MovAvg()
        lossL = []

        n_won = 0                   # number of wins while training
        n_terminals = 0             # number of terminal states reached while training
        last_terminals = 0          # previous number of terminal states

        succeeded_row_curr = 0      # current number of succeeded tests in a row
        succeeded_row_max = 0       # max number of succeeded tests in a row

        batch_ix = 1 # index starts from 1
        stime = time.time()
        while batch_ix <= num_batches:

            inspect_now = inspect and batch_ix % test_freq == 0

            ### play for a batch of data

            observations, actions, rewards, next_observations, terminals, wons = self.run_play(
                steps=          self.batch_size,
                training=       True,
                reset=          False,
                break_terminal= False,
                inspect=        inspect_now)

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

            self._publish(batch=batch, training_data=training_data, metrics=metrics, inspect=inspect)
            self._upd_step += 1

            ### test

            if batch_ix % test_freq == 0:

                tr_nfo = f'# {batch_ix:3} term:{n_terminals}(+{n_terminals - last_terminals}) loss:{loss_mavg():.4f}'
                last_terminals = n_terminals

                # single episode
                _, actions, rewards, _, _, wons = self.run_play(
                    steps=          test_max_steps,
                    training=       False,
                    reset=          True,
                    break_terminal= True,
                    inspect=        inspect)
                ts_one_nfo = f'1TS: {len(actions)} actions, return {sum(rewards):.1f} ({"won" if sum(wons) else "lost"})'

                # few tests
                avg_won, avg_return = self.test_on_episodes(
                    n_episodes= test_episodes,
                    max_steps=  test_max_steps)
                ts_nfo = f'{test_episodes}xTS: avg_won:{avg_won * 100:.1f}%, avg_return:{avg_return:.1f}'

                if avg_won == 1:
                    succeeded_row_curr += 1
                    if succeeded_row_curr > succeeded_row_max: succeeded_row_max = succeeded_row_curr
                else:
                    succeeded_row_curr = 0

                self._rlog.info(f'{tr_nfo} -- {ts_one_nfo} -- {ts_nfo}')

                # INFO: Envy should be reset after tests and already will be since every test run ends here with terminal

            batch_ix += 1

            if break_ntests is not None and succeeded_row_curr == break_ntests: break

        self._rlog.info(f'### Training finished, time taken: {time.time() - stime:.2f}sec')

        return {
            'n_actions':            (batch_ix-1)*self.batch_size,   # total number of training actions
            'lossL':                lossL,
            'n_terminals':          n_terminals,
            'n_won':                n_won,
            'n_updates_done':       batch_ix-1,
            'succeeded_row_max':    succeeded_row_max}

    # extracts data from a batch + eventually adds new
    @abstractmethod
    def _build_training_data(self, batch:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]: pass

    # updates policy or value function (Actor, Critic ar any other component), returns metrics with 'loss'
    @abstractmethod
    def _update(self, training_data:Dict[str,np.ndarray]) -> Dict[str,Any]: pass

    # plays n episodes, returns tuple with won_factor & avg_reward
    def test_on_episodes(
            self,
            n_episodes: int=            10,
            max_steps: Optional[int]=   None,
    ) -> Tuple[float, float]:
        n_won = 0
        sum_rewards = 0
        for e in range(n_episodes):
            observations, _, rewards, _, _, wons = self.run_play(
                steps=          max_steps,
                training=       False,
                reset=          True,
                break_terminal= True,
                inspect=        False)
            n_won += sum(wons)
            sum_rewards += sum(rewards)
        return n_won/n_episodes, sum_rewards/n_episodes

    # publishes to TB / inspects data
    def _publish(
            self,
            batch: Dict[str,np.ndarray],
            training_data: Dict[str,np.ndarray],
            metrics: Dict[str,Any],
            inspect: bool,
    ) -> None:

        if self._tbwr:
            self._tbwr.add(value=metrics['loss'], tag=f'actor/loss', step=self._upd_step)

        if inspect:
            plot_obs_act(observations=batch['observations'], actions=batch['actions'])
            plot_rewards(rewards=batch['rewards'])

    # returns Actor save directory
    def get_save_dir(self) -> str:
        return f'{self.save_topdir}/{self.name}'

    @abstractmethod
    def save(self): pass

    @abstractmethod
    def load(self): pass

    def __str__(self):
        nfo =  f'{self.__class__.__name__} (TrainableActor) : {self.name}\n'
        nfo += f'> observation width: {self._observation_vector(self.envy.get_observation()).shape[-1]}'
        return nfo