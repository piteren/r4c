from abc import abstractmethod, ABC
import numpy as np
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.moving_average import MovAvg
import time
from torchness.tbwr import TBwr
from torchness.comoneural.avg_probs import avg_mm_probs
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import List, Tuple, Optional, Dict, Any, Union

from r4c.envy import RLEnvy, FiniteActionsRLEnvy
from r4c.actor import TrainableActor
from r4c.helpers import RLException, NUM


# Trainer Experience Memory
class ExperienceMemory:

    def __init__(
            self,
            max_size: Optional[int],
            seed: int):
        self._mem: Dict[str,np.ndarray] = {}
        self._init_mem()
        self.max_size = max_size
        np.random.seed(seed)


    def _init_mem(self):
        self._mem = {
            'observations':         None, # np.ndarray of NUM (2 dim)
            'actions':              None, # np.ndarray of ints
            'rewards':              None, # np.ndarray of floats
            'next_observations':    None, # np.ndarray of NUM (2 dim)
            'terminals':            None, # np.ndarray of bool
            'wons':                 None} # np.ndarray of bool

    # adds given experience
    def add(self, experience:Dict[str, Union[List,np.ndarray]]):

        # add or put
        for k in experience:
            ex_np = np.asarray(experience[k])
            if self._mem[k] is None:
                self._mem[k] = ex_np
            else:
                self._mem[k] = np.concatenate([self._mem[k], ex_np])

        # trim if needed
        if self.max_size and len(self) > self.max_size:
            for k in self._mem:
                self._mem[k] = self._mem[k][-self.max_size:]

    # returns random sample of non-duplicates from memory
    def get_sample(self, n:int) -> Dict[str,np.ndarray]:
        ixs = np.random.choice(len(self), n, replace=False)
        return {k: self._mem[k][ixs] for k in self._mem}


    # returns (copy) of full memory
    def get_all(self, reset=True) -> Dict[str,np.ndarray]:
        mc = {k: np.copy(self._mem[k]) for k in self._mem}
        if reset: self._init_mem()
        return mc


    def clear(self):
        self._init_mem()


    def __len__(self):
        return len(self._mem['observations']) if self._mem['observations'] is not None else 0


# Reinforcement Learning Trainer for Actor acting on RLEnvy
class RLTrainer(ABC):
    """
    Implements generic RL training procedure -> train()
    This procedure is valid for some RL algorithms (QTable, PG, AC)
    and may be overridden with custom implementation.
    """

    def __init__(
            self,
            envy: RLEnvy,
            actor: TrainableActor,
            batch_size: int,                    # Actor update batch data size
            exploration: float,                 # train exploration factor
            train_sampled: float,               # how often move is sampled (vs argmax) while training
            mem_batches: Optional[int]= None,   # ExperienceMemory max size (in number of batches)
            seed: int=                  123,
            logger=                     None,
            loglevel=                   20,
            hpmser_mode=                False):

        self._rlog = logger or get_pylogger(level=loglevel)
        self._rlog.info(f'*** RLTrainer *** initializes..')
        self._rlog.info(f'> Envy:            {envy.__class__.__name__}')
        self._rlog.info(f'> Actor:           {actor.__class__.__name__}, name: {actor.name}')
        self._rlog.info(f'> batch_size:      {batch_size}')
        self._rlog.info(f'> memory max size: {batch_size*mem_batches if mem_batches else "None"}')
        self._rlog.info(f'> exploration:     {exploration}')
        self._rlog.info(f'> train_sampled:   {train_sampled}')
        self._rlog.info(f'> seed:            {seed}')

        self.envy = envy
        self.actor = actor
        self.batch_size = batch_size
        self.mem_max_size = self.batch_size * mem_batches if mem_batches else None
        self.exploration = exploration
        self.train_sampled = train_sampled
        self.memory: Optional[ExperienceMemory] = None
        self.seed = seed
        np.random.seed(self.seed)
        self.hpmser_mode = hpmser_mode

        self._tbwr = TBwr(logdir=self.actor.get_save_dir()) if not self.hpmser_mode else None
        self._upd_step = 0 # global Trainer update step

        self._zepro = ZeroesProcessor(
            intervals=  (10,50,100),
            tbwr=       self._tbwr) if not self.hpmser_mode else None

    # plays (envy) until N steps performed or terminal state, collects and returns experience
    def play(
            self,
            reset: bool,            # for True starts play from the initial state
            steps: int,
            break_terminal: bool,   # for True breaks play at terminal state
            exploration: float,
            sampled: float,
            render: bool,
    ) -> Tuple[
        List[np.ndarray],   # observations
        List[NUM],          # actions
        List[float],        # rewards
        List[bool],         # terminals
        List[bool],         # wons
    ]:

        self._rlog.log(5,f'playing for {steps} steps..')

        if reset: self.envy.reset()

        observations = []
        actions = []
        rewards = []
        terminals = []
        wons = []

        while len(actions) < steps:

            observation, action, reward = self._exploratory_move(
                exploration=    exploration,
                sampled=        sampled)

            is_terminal = self.envy.is_terminal()
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(is_terminal)
            wons.append(self.envy.won())

            if render: self.envy.render()

            if is_terminal and break_terminal:
                break

        self._rlog.log(5,f'played {len(actions)} steps (break_terminal is {break_terminal})')
        return observations, actions, rewards, terminals, wons

    # plays one episode from reset till terminal state or max_steps
    def _play_episode(
            self,
            exploration: float,
            sampled: float,
            render: bool,
            max_steps: Optional[int]=   None,  # if max steps is given then single play for max_steps is considered to be won
    ) -> Tuple[
        List[np.ndarray],   # observations
        List[NUM],          # actions
        List[float],        # rewards
        bool,               # won
    ]:

        if max_steps is None and self.envy.get_max_steps() is None:
            raise RLException('Cannot play episode for Envy where max_steps is None and given max_steps is None')

        observations, actions, rewards, terminals, wons = self.play(
            reset=          True,
            steps=          max_steps or self.envy.get_max_steps(),
            break_terminal= True,
            exploration=    exploration,
            sampled=        sampled,
            render=         render)

        return observations, actions, rewards, wons[-1]

    # performs one Actor move (observation -> action -> reward)
    def _exploratory_move(
            self,
            exploration=    0.0, # prob pf exploration
            sampled=        0.0, # prob of sampling (vs argmax)
    ) -> Tuple[
        np.ndarray, # observation
        NUM,        # action
        float,      # reward
    ]:

        # reset Envy if needed
        if self.envy.is_terminal():
            self.envy.reset()

        # prepare observation vector
        pre_action_observation = self.envy.get_observation()
        obs_vec = self.actor.observation_vector(pre_action_observation)

        # get and run action
        if np.random.rand() < exploration: action = self._get_exploring_action()
        else:                              action = self.actor.get_policy_action(
                                                        observation=    obs_vec,
                                                        sampled=        np.random.rand() < sampled)
        reward = self.envy.run(action)

        return obs_vec, action, reward

    # trainer selects exploring action (with Trainer exploratory policy)
    @abstractmethod
    def _get_exploring_action(self) -> NUM: pass

    # generic RL training procedure, returns dict with some training stats
    def train(
            self,
            num_updates: int,                       # number of training updates
            upd_on_episode=                 False,  # updates on episode finish / terminal (does not wait till batch)
            test_freq=                      100,    # number of updates between test
            test_episodes: int=             100,    # number of testing episodes
            test_max_steps: Optional[int]=  None,   # max number of episode steps while testing
            test_render: bool=              False,  # renders one episode while test
            inspect: bool=                  False,  # for debug / research
            break_ntests: Optional[int]=    None,   # breaks training after all test episodes succeeded N times in a row
    ) -> dict:

        stime = time.time()
        self._rlog.info(f'Starting train for {num_updates} updates..')

        self.memory = ExperienceMemory(
            max_size=   self.mem_max_size,
            seed=       self.seed)
        self._rlog.info(f'> initialized ExperienceMemory of maxsize {self.mem_max_size}')

        self.envy.reset()
        loss_mavg = MovAvg()
        lossL = []
        n_actions = 0               # total number of train actions
        n_terminals = 0             # number of terminal states reached while training
        last_terminals = 0          # previous number of terminal states
        n_won = 0                   # number of wins while training
        succeeded_row_curr = 0      # current number of succeeded tests in a row
        succeeded_row_max = 0       # max number of succeeded tests in a row
        for upd_ix in range(num_updates):

            # get a batch of data
            n_batch_actions = 0
            while n_batch_actions < self.batch_size:

                # plays till episode end, to allow update on episode
                observations, actions, rewards, terminals, wons = self.play(
                    steps=          self.batch_size - n_batch_actions,
                    reset=          False,
                    break_terminal= True,
                    exploration=    self.exploration,
                    sampled=        self.train_sampled,
                    render=         False)

                na = len(actions)
                n_batch_actions += na
                n_actions += na

                last_obs = self.actor.observation_vector(self.envy.get_observation())
                next_observations = observations[1:] + [last_obs]

                # INFO: not all algorithms (QLearning,PG,AC) need all the data below (we store 'more' just in case)
                self.memory.add(experience={
                    'observations':         observations,
                    'actions':              actions,
                    'rewards':              rewards,
                    'next_observations':    next_observations,
                    'terminals':            terminals,
                    'wons':                 wons})

                if terminals[-1]:
                    n_terminals += 1 # ..may not be terminal when limit of n_batch_actions reached
                if wons[-1]:
                    n_won += 1

                self._rlog.debug(f' >> Trainer gots {len(observations):3} observations after play and {len(self.memory):3} in memory, n_batch_actions: {n_batch_actions}' )

                if upd_on_episode: break

            # update Actor & process metrics
            upd_metrics = self._update_actor(inspect=inspect and upd_ix % test_freq == 0)
            self._upd_step += 1

            if 'loss' in upd_metrics: lossL.append(loss_mavg.upd(upd_metrics['loss']))

            # process / monitor policy probs
            if self._tbwr and 'probs' in upd_metrics:
                for k,v in avg_mm_probs(upd_metrics.pop('probs')).items():
                    self._tbwr.add(value=v, tag=f'actor_upd/{k}', step=self._upd_step)

            if self._zepro and 'zeroes' in upd_metrics:
                self._zepro.process(zs=upd_metrics.pop('zeroes'))

            if self._tbwr:
                for k,v in upd_metrics.items():
                    #if k not in ['value','advantage','qvs']: # TODO <- those are here as Tensors with shape [256] - not value - not Ok for TB <- fix it
                    #print(v.shape)
                    #print(k,v)
                    self._tbwr.add(value=v, tag=f'actor_upd/{k}', step=self._upd_step)

            # test Actor
            if upd_ix % test_freq == 0:

                # single episode
                observations, actions, rewards, won = self._play_episode(
                    exploration=    0.0,
                    sampled=        0.0,
                    render=         test_render,
                    max_steps=      test_max_steps)

                # few tests
                avg_won, avg_return = self.test_on_episodes(
                    n_episodes=     test_episodes,
                    max_steps=      test_max_steps)

                self._rlog.info(f'# {upd_ix:3} term:{n_terminals}(+{n_terminals-last_terminals}) -- TS: {len(actions)} actions, return {sum(rewards):.1f} ({"won" if won else "lost"}) -- {test_episodes}xTS: avg_won: {avg_won*100:.1f}%, avg_return: {avg_return:.1f} -- loss_actor: {loss_mavg():.4f}')
                last_terminals = n_terminals

                if avg_won == 1:
                    succeeded_row_curr += 1
                    if succeeded_row_curr > succeeded_row_max: succeeded_row_max = succeeded_row_curr
                else: succeeded_row_curr = 0

            if break_ntests is not None and succeeded_row_curr==break_ntests: break

        self._rlog.info(f'### Training finished, time taken: {time.time()-stime:.2f}sec')

        return { # training_report
            'n_actions':            n_actions,
            'lossL':                lossL,
            'n_terminals':          n_terminals,
            'n_won':                n_won,
            'succeeded_row_max':    succeeded_row_max}

    # updates Actor policy, returns dict with Actor "metrics" (check Actor.update_with_experience())
    def _update_actor(self, inspect:bool=False) -> Dict[str,Any]:
        batch = self.memory.get_sample(self.batch_size)
        return self.actor.update_with_experience(
            batch=      batch,
            inspect=    inspect)

    # plays n episodes, returns (won_factor, avg/reward)
    def test_on_episodes(
            self,
            n_episodes=                 100,
            max_steps: Optional[int]=   None,
    ) -> Tuple[float, float]:
        n_won = 0
        sum_rewards = 0
        for e in range(n_episodes):
            observations, actions, rewards, won = self._play_episode(
                exploration=    0.0,
                sampled=        0.0,
                render=         False,
                max_steps=      max_steps)
            n_won += int(won)
            sum_rewards += sum(rewards)
        return n_won/n_episodes, sum_rewards/n_episodes


# FiniteActions RL Trainer (for Actor acting on FiniteActionsRLEnvy)
class FATrainer(RLTrainer, ABC):

    def __init__(self, envy:FiniteActionsRLEnvy, seed:int, **kwargs):
        np.random.seed(seed)
        RLTrainer.__init__(self, envy=envy, seed=seed, **kwargs)
        self.envy = envy  # INFO: type "upgrade" for pycharm editor

        self._rlog.info(f'*** FATrainer *** initialized')
        self._rlog.info(f'> number of actions: {self.envy.num_actions()}')

    # selects 100% random action from action space, (np. seed is fixed at Trainer)
    def _get_exploring_action(self) -> int:
        return int(np.random.choice(self.envy.num_actions()))