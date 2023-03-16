import numpy as np
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.moving_average import MovAvg
import time
from typing import List, Tuple, Optional, Dict, Union

from r4c.envy import RLEnvy
from r4c.actor import TrainableActor
from r4c.helpers import RLException, NUM, plot_obs_act, plot_rewards


# Runner Experience Memory
class ExperienceMemory:
    """
    Experience Memory stores data generated while Actor plays with its policy on Envy.
    Data is stored as numpy arrays.
    """

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
            'actions':              None, # np.ndarray of NUM
            'rewards':              None, # np.ndarray of floats
            'next_observations':    None, # np.ndarray of NUM (2 dim)
            'terminals':            None, # np.ndarray of bool
            'wons':                 None} # np.ndarray of bool

    # adds given experience
    def add(self, experience:Dict[str,Union[List,np.ndarray]]):

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


# Reinforcement Learning Runner runs TrainableActor on RLEnvy
class RLRunner:

    def __init__(
            self,
            envy: RLEnvy,
            actor: TrainableActor,
            seed: int=                  123,
            logger=                     None,
            loglevel=                   20):

        self._rlog = logger or get_pylogger(level=loglevel)
        self._rlog.info(f'*** RLRunner *** initializes..')
        self._rlog.info(f'> Envy: {envy.__class__.__name__}')
        self._rlog.info(f'> Actor: {actor.name} ({actor.__class__.__name__})')

        self.envy = envy
        self.actor = actor
        self.memory: Optional[ExperienceMemory] = None
        self.seed = seed
        np.random.seed(self.seed)

    # plays Actor on Envy, collects and returns experience
    def play(
            self,
            reset: bool,            # for True starts play from the initial state
            steps: int,             # number of steps to play
            break_terminal: bool,   # for True breaks play at terminal state
            exploration: float,
            sampled: float,
            inspect: bool,
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

            observation, action, reward = self._move(
                exploration=    exploration,
                sampled=        sampled)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(self.envy.is_terminal())
            wons.append(self.envy.won())

            if inspect: self.envy.render()

            if terminals[-1] and break_terminal:
                break

        if inspect:
            plot_obs_act(observations=observations, actions=actions)
            kw = {'rewards': rewards}
            if 'discount' in self.actor.__dict__:
                kw.update({
                    'terminals':        terminals,
                    'discount':         self.actor.__dict__['discount'],
                    'movavg_factor':    self.actor.__dict__['movavg_factor'],
                })
            plot_rewards(**kw)

        self._rlog.log(5,f'played {len(actions)} steps (break_terminal is {break_terminal})')
        return observations, actions, rewards, terminals, wons

    # performs one Actor move (observation -> action -> reward)
    def _move(
            self,
            exploration: float, # exploration factor (probability)
            sampled: float,     # sampling (vs argmax) factor (probability)
    ) -> Tuple[
        np.ndarray, # observation
        NUM,        # action
        float,      # reward
    ]:

        # eventually reset Envy
        if self.envy.is_terminal():
            self.envy.reset()

        # prepare observation vector
        pre_action_observation = self.envy.get_observation()
        observation_vector = self.actor.observation_vector(pre_action_observation)

        # get and run action
        action = self.actor.get_action(
            observation=    observation_vector,
            explore=        np.random.rand() < exploration,
            sample=         np.random.rand() < sampled)
        reward = self.envy.run(action)

        return observation_vector, action, reward

    # plays one episode from reset till terminal state or max_steps
    def play_episode(
            self,
            max_steps: Optional[int]=   None,
            exploration: float=         0.0,
            sampled: float=             0.0,
            inspect: bool=              False,
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
            inspect=        inspect)

        return observations, actions, rewards, wons[-1]

    # RL training procedure, returns dict with some training stats
    def train(
            self,
            num_updates: int,                       # number of training updates
            batch_size: int,                        # Actor update batch data size
            mem_batches: Optional[int]=     None,   # ExperienceMemory max size (in number of batches)
            sample_memory: bool=            False,  # sample batch from memory or get_all and reset
            exploration: float=             0.0,    # exploration probability while building experience
            sampled_TR: float=              0.0,    # sampling probability while building experience
            sampled_PL: float=              0.0,    # sampling probability while playing
            upd_on_episode=                 False,  # updates on episode finish / terminal (does not wait till batch)
            test_freq=                      100,    # number of updates between test
            test_episodes: int=             100,    # number of testing episodes
            test_max_steps: Optional[int]=  None,   # max number of episode steps while testing
            inspect: bool=                  False,  # debug / research, inspects data while updating Actor and testing
            break_ntests: Optional[int]=    None,   # breaks training after all test episodes succeeded N times in a row
    ) -> dict:
        """
        Generic RL train procedure.
        This procedure is valid for some RL algorithms (QTable, PG, AC),
        its is build with a loop where Actor:
        1. plays on Envy to get a batch of experience data (stored in memory)
        2. is updated (policy or Value function) and returns some metrics
        3. is tested (evaluated) on Environment
        """

        self._rlog.info(f'Starting train for {num_updates} updates..')

        mem_max_size = batch_size * mem_batches if mem_batches is not None else None
        self.memory = ExperienceMemory(
            max_size=   mem_max_size,
            seed=       self.seed)
        self._rlog.info(f'> initialized ExperienceMemory of max size {mem_max_size}')

        self.envy.reset()

        loss_mavg = MovAvg()
        lossL = []
        n_actions = 0               # total number of train actions
        n_terminals = 0             # number of terminal states reached while training
        last_terminals = 0          # previous number of terminal states
        n_won = 0                   # number of wins while training
        succeeded_row_curr = 0      # current number of succeeded tests in a row
        succeeded_row_max = 0       # max number of succeeded tests in a row

        stime = time.time()
        for upd_ix in range(num_updates):

            ### 1. get a batch of data (play)

            n_batch_actions = 0
            while n_batch_actions < batch_size:

                # plays till episode end, to allow update on episode
                observations, actions, rewards, terminals, wons = self.play(
                    steps=          batch_size - n_batch_actions,
                    reset=          False,
                    break_terminal= True,
                    exploration=    exploration,
                    sampled=        sampled_TR,
                    inspect=        False)

                na = len(actions)
                n_batch_actions += na
                n_actions += na

                last_obs = self.actor.observation_vector(self.envy.get_observation())
                next_observations = observations[1:] + [last_obs]

                # store experience data used to update Actor
                self.memory.add(experience={
                    'observations':         observations,
                    'actions':              actions,
                    'rewards':              rewards,
                    'next_observations':    next_observations,
                    'terminals':            terminals,
                    'wons':                 wons})

                if terminals[-1]: n_terminals += 1
                if wons[-1]:      n_won += 1

                self._rlog.debug(f' >> Trainer gots {len(observations):3} observations after play and {len(self.memory):3} in memory, n_batch_actions: {n_batch_actions}' )

                if upd_on_episode: break

            ### 2. update an Actor & process metrics

            batch = self.memory.get_sample(batch_size) if sample_memory else self.memory.get_all()
            loss = self.actor.update_with_experience(
                batch=      batch,
                inspect=    inspect and upd_ix % test_freq == 0)
            lossL.append(loss_mavg.upd(loss))

            """
            # process / monitor policy probs
            if self._tbwr and 'probs' in upd_metrics:
                for k,v in avg_mm_probs(upd_metrics.pop('probs')).items():
                    self._tbwr.add(value=v, tag=f'actor_upd/{k}', step=self._upd_step)

            if self._zepro and 'zeroes' in upd_metrics:
                self._zepro.process(zs=upd_metrics.pop('zeroes'))

            if self._tbwr:
                for k,v in upd_metrics.items():
                    self._tbwr.add(value=v, tag=f'actor_upd/{k}', step=self._upd_step)
            """

            ### 3. test an Actor

            if upd_ix % test_freq == 0:

                # single episode
                observations, actions, rewards, won = self.play_episode(
                    max_steps=      test_max_steps,
                    exploration=    0.0,
                    sampled=        sampled_PL,
                    inspect=        inspect)

                # few tests
                avg_won, avg_return = self.test_on_episodes(
                    n_episodes=     test_episodes,
                    sampled=        sampled_PL,
                    max_steps=      test_max_steps)

                self._rlog.info(f'# {upd_ix:3} term:{n_terminals}(+{n_terminals-last_terminals}) -- TS: {len(actions)} actions, return {sum(rewards):.1f} ({"won" if won else "lost"}) -- {test_episodes}xTS: avg_won: {avg_won*100:.1f}%, avg_return: {avg_return:.1f} -- loss_actor: {loss_mavg():.4f}')
                last_terminals = n_terminals

                if avg_won == 1:
                    succeeded_row_curr += 1
                    if succeeded_row_curr > succeeded_row_max: succeeded_row_max = succeeded_row_curr
                else: succeeded_row_curr = 0

            if break_ntests is not None and succeeded_row_curr==break_ntests: break

        self._rlog.info(f'### Training finished, time taken: {time.time()-stime:.2f}sec')

        return {
            'n_actions':            n_actions,
            'lossL':                lossL,
            'n_terminals':          n_terminals,
            'n_won':                n_won,
            'succeeded_row_max':    succeeded_row_max}

    # plays n episodes, returns (won_factor, avg/reward)
    def test_on_episodes(
            self,
            n_episodes=                 100,
            sampled=                    0.0,
            max_steps: Optional[int]=   None,
    ) -> Tuple[float, float]:
        n_won = 0
        sum_rewards = 0
        for e in range(n_episodes):
            observations, actions, rewards, won = self.play_episode(
                max_steps=      max_steps,
                exploration=    0.0,
                sampled=        sampled,
                inspect=        False)
            n_won += int(won)
            sum_rewards += sum(rewards)
        return n_won/n_episodes, sum_rewards/n_episodes