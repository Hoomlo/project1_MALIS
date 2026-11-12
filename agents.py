"""Agents for Fishing Season (MALIS, Project 1).

This file is a skeleton. Write all your agents in it.

Do NOT modify the classes Agent and ContextualAgent, nor the function play():
they define how every agent interacts with the simulator, and they are used
to run your agents during grading.

How an agent works
------------------
For each season, play(agent, bay) calls:

    agent.reset()                     once, before the first day
    spot = agent.choose()             every day: return a spot, 0 to 4
    agent.update(spot, catch)         every day: the catch you observed

An agent only learns through update(). It must not look at the bay itself.

Contextual agents (Part IV, optional) follow the same pattern, but receive the
morning conditions x (numpy array [wind_kmh, waves_m, tide_m]):

    spot = agent.choose(x)
    agent.update(x, spot, catch)

Example
-------
    from fishing import FishingBay
    from agents import RandomAgent, play

    bay = FishingBay(season=1, seed=0)
    total = play(RandomAgent(seed=0), bay)
"""
import numpy as np

from fishing import ContextualBay, FishingBay

# ---------------------------------------------------------------------------
# Provided: do not modify
# ---------------------------------------------------------------------------


class Agent:
    """Base class for the agents of Parts II and III.

    Parameters
    ----------
    n_spots : int
        Number of fishing spots (5 in the game).
    seed : int or None
        Seed of the agent's own random generator, self.rng. Use self.rng for
        any random decision, so that results can be reproduced.
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self):
        """Return the spot to fish today, an int between 0 and n_spots - 1."""
        raise NotImplementedError

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed (kg)."""


class ContextualAgent:
    """Base class for the agents of Part IV (contextual bay).

    Same as Agent, except that choose() and update() also receive x, the
    conditions of the day: numpy array [wind_kmh, waves_m, tide_m].
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self, x):
        """Return the spot to fish today given the conditions x."""
        raise NotImplementedError

    def update(self, x, spot, catch):
        """Called after each day with the conditions, the spot chosen and the catch observed."""


def play(agent, bay):
    """Play one full season of bay with agent. Returns the total catch (kg).

    Works with FishingBay and Agent, or with ContextualBay and ContextualAgent.
    The bay is reset first, so the same bay can be played by several agents.
    """
    contextual = isinstance(bay, ContextualBay)
    if contextual and not isinstance(agent, ContextualAgent):
        raise TypeError("a ContextualBay must be played by a ContextualAgent")
    if not contextual and not isinstance(agent, Agent):
        raise TypeError("a FishingBay must be played by an Agent")
    bay.reset()
    agent.reset()
    while not bay.done:
        if contextual:
            x = bay.observe_array()
            spot = int(agent.choose(x))
            catch = bay.fish(spot)
            agent.update(x, spot, catch)
        else:
            spot = int(agent.choose())
            catch = bay.fish(spot)
            agent.update(spot, catch)
    return bay.total


# ---------------------------------------------------------------------------
# Provided example agents (baselines)
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Chooses a spot uniformly at random every day. Learns nothing."""

    def choose(self):
        return int(self.rng.integers(self.n_spots))


class RandomContextualAgent(ContextualAgent):
    """Ignores the conditions and chooses a spot uniformly at random every day."""

    def choose(self, x):
        return int(self.rng.integers(self.n_spots))


# ---------------------------------------------------------------------------
# Your agents
# ---------------------------------------------------------------------------


class HumanRuleAgent(Agent):
    """Task 4: the best human strategy identified in Part II, written as a rule.

    Describe the rule here, precisely enough that someone could apply it by hand.
    Add constructor parameters if your rule has any (call super().__init__ first).

    Rule : Go to each spot, keep the better spot and then fish on this exact spot 
    until the season ends
    """
    def __init__(self, n_spots=5, seed=None):
        super().__init__(n_spots, seed)
        self.day = 0
        self.catches = np.zeros(n_spots)
        self.best_spot = 0
    
    def reset(self):
        self.day = 0
        self.catches = np.zeros(self.n_spots)
        self.best_spot = 0

    def choose(self):
        if self.day < self.n_spots :
            return self.day
        else :
            return self.best_spot

    def update(self, spot, catch):
        if self.day < self.n_spots : 
            self.catches[spot] = catch
            #If we reach the end of the unexeplored spot, we keep the best one
            if self.day == self.n_spots - 1:
                self.best_spot = int(np.argmax(self.catches))
        self.day += 1


# Task 4: add below at least two agents implementing methods from the reading
# (Sutton and Barto, Chapter 2), as subclasses of Agent. Document each class:
# what it keeps in memory, how it chooses, what its parameters mean.
class UCBAgent(Agent):
    """Upper Confidence Bound (UCB) Bandit Algorithm.
    
    Parameters
    ----------
    c : float
        Degree of exploration.
    """
    def __init__(self, n_spots=5, seed=None, c=2.0):
        super().__init__(n_spots, seed)
        self.c = c
        self.q_values = np.zeros(n_spots)
        self.action_counts = np.zeros(n_spots)
        self.t = 0

    def reset(self):
        self.q_values = np.zeros(self.n_spots)
        self.action_counts = np.zeros(self.n_spots)
        self.t = 0

    def choose(self):
        # Force testing every spot at least once to avoid dividing by zero
        if 0 in self.action_counts:
            return int(np.argmin(self.action_counts))
        
        # UCB formula from the book
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.action_counts)
        return int(np.argmax(ucb_values))

    def update(self, spot, catch):
        self.t += 1
        self.action_counts[spot] += 1
        self.q_values[spot] += (catch - self.q_values[spot]) / self.action_counts[spot]

class EpsilonGreedyAgent(Agent):
    """
    Sutton and Barto, Section 2.2: epsilon-greedy action selection.
    Section 2.5: Tracking a Nonstationary Problem (using constant alpha).
    
    Parameters:
    - epsilon (float): probability of choosing a random action (exploration).
    - alpha (float or None): step-size parameter for the update. If None, 
      uses the sample average (1/N).
    
    Memory:
    - self.q_values (array): value estimation (catch) for each spot.
    - self.counts (array): number of times each spot has been fished.
    """
    def __init__(self, n_spots=5, seed=None, epsilon=0.1, alpha=None):
        super().__init__(n_spots, seed)
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.zeros(n_spots)
        self.counts = np.zeros(n_spots)

    def reset(self):
        self.q_values = np.zeros(self.n_spots)
        self.counts = np.zeros(self.n_spots)

    def choose(self):
        # Exploration with probability epsilon
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_spots))
        # Exploitation (greedy choice) with probability 1 - epsilon
        else:
            # np.argmax returns the index of the max value (the best spot)
            return int(np.argmax(self.q_values))

    def update(self, spot, catch):
        self.counts[spot] += 1
        
        # Section 2.4: Sample-average method
        if self.alpha is None:
            step_size = 1.0 / self.counts[spot]
        # Section 2.5: Constant step-size (ideal for Season 3)
        else:
            step_size = self.alpha
            
        # Update formula: Q(a) <- Q(a) + step_size * [Reward - Q(a)]
        self.q_values[spot] += step_size * (catch - self.q_values[spot])


# Task 6 (optional): add your contextual agents below, as subclasses of
# ContextualAgent.
