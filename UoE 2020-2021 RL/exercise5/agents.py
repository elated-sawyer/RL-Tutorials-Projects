from abc import ABC, abstractmethod
from collections import defaultdict
import random
import sys
from typing import List, Dict, DefaultDict

import numpy as np
from gym.spaces import Space, Box
from gym.spaces.utils import flatdim

from rl2021.exercise5.matrix_game import actions_to_onehot

def obs_to_tuple(obs):
    return tuple([tuple(o) for o in obs])


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        observation_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param observation_spaces (List[Space]): observation spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self, obs: List[np.ndarray]) -> List[int]:
        """Chooses an action for all agents given observations

        :param obs (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents


        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            act_vals = [self.q_tables[i][(obss[i], act)] for act in range(self.n_acts[i])]
            max_val = max(act_vals)
            max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
            # the set of acts, that has max Q
            if random.random() < self.epsilon:
                actions.append(random.randint(0, self.n_acts[i] - 1))
            else:
                actions.append(random.choice(max_acts))
        return actions


    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            # Compute Q targets for current states
            A = np.zeros(self.n_acts[i])
            for k in range(self.n_acts[i]):
                A[k] = self.q_tables[i][(n_obss[i], k)]
            q_targets = rewards[i] + self.gamma * (1 - dones[i]) * max(A)
            q_expected = self.q_tables[i][(obss[i], actions[i])]
            self.q_tables[i][(obss[i], actions[i])] += float(self.learning_rate * (q_targets - q_expected))
            updated_values.append(self.q_tables[i][(obss[i], actions[i])])
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        self.epsilon = 1.0-(min(1.0, timestep/(0.07*max_timestep)))*0.95


class JointActionLearning(MultiAgent):
    """Agents using the Joint Action Learning algorithm with Opponent Modelling

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of
            observations and joint actions to respective Q-values for all agents
        :attr models (List[DefaultDict[DefaultDict]]): each agent holding model of other agent
            mapping observation to other agent actions to count of other agent action

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount
        rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: defaultdict(lambda: 0)) for _ in range(self.num_agents)] 

        # count observations - count for each agent
        self.c_obss = [defaultdict(lambda: 0) for _ in range(self.num_agents)]


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            EV = []
            if random.random() < self.epsilon:
                joint_action.append(random.randint(0, self.n_acts[i] - 1))
            else:
                if self.c_obss[i][obss[i]] == 0.0:
                    joint_action.append(random.randint(0, self.n_acts[i] - 1))
                else:
                    if i == 0:
                        for ai in range(self.n_acts[i]):
                            EV.append(sum([self.models[i][obss[i]][a_i] / self.c_obss[i][obss[i]] * self.q_tables[i][
                                (obss[i], (ai, a_i))] for a_i in range(self.n_acts[i])]))
                    else:
                        for ai in range(self.n_acts[i]):
                            EV.append(sum([self.models[i][obss[i]][a_i] / self.c_obss[i][obss[i]] * self.q_tables[i][
                                (obss[i], (a_i, ai))] for a_i in range(self.n_acts[i])]))
                    joint_action.append(np.argmax(EV))
        return joint_action


    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            EV = []
            if i == 0:
                if self.c_obss[i][obss[i]] == 0.0:
                    for ai in range(self.n_acts[i]):
                        EV.append(sum([self.q_tables[i][(obss[i], (ai, a_i))] / self.n_acts[i] for a_i in
                                       range(self.n_acts[i])]))
                else:
                    for ai in range(self.n_acts[i]):
                        EV.append(sum([self.models[i][obss[i]][a_i] / self.c_obss[i][obss[i]] * self.q_tables[i]
                        [(obss[i], (ai, a_i))] for a_i in range(self.n_acts[i])]))
            else:
                if self.c_obss[i][obss[i]] == 0.0:
                    for ai in range(self.n_acts[i]):
                        EV.append(sum([self.q_tables[i][(obss[i], (a_i, ai))] / self.n_acts[i] for a_i in
                                       range(self.n_acts[i])]))
                else:
                    for ai in range(self.n_acts[i]):
                        EV.append(sum([self.models[i][obss[i]][a_i] / self.c_obss[i][obss[i]] * self.q_tables[i]
                        [(obss[i], (a_i, ai))] for a_i in range(self.n_acts[i])]))
            q_targets = rewards[i] + self.gamma * (1 - dones[i]) * max(EV)
            q_expected = self.q_tables[i][(obss[i], (actions[0], actions[1]))]
            self.q_tables[i][(obss[i], (actions[0], actions[1]))] += float(
                self.learning_rate * (q_targets - q_expected))

            self.c_obss[i][obss[i]] += 1
            self.models[i][obss[i]][actions[-(i + 1)]] += 1

            updated_values.append(self.q_tables[i][(
            obss[i], (actions[i], actions[-(i + 1)]))])  ##########update???, different agent in the same list???????
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        self.epsilon = 1.0 - (min(1.0, timestep / (0.07 * max_timestep))) * 0.95
