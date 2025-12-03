# qlearningAgents.py


from game import *
from agents.learningAgents import ReinforcementAgent
from pacman import GameState

import random,util,math
import numpy as np
from game import Directions
import json


class Q2Agent(ReinforcementAgent):
    """
      Q-Learning Agent

      Methods you should fill in:
        - __init__
        - registerInitialState
        - update
        - getParams
        - epsilonGreedyActionSelection

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
    """

    def __init__(self, usePresetParams=False, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p Q2Agent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes

        usePresetParams - Set to True if you want to use your evaluation parameters
        """

        self.index = 0  # This is always Pacman

        ReinforcementAgent.__init__(self, **args)

        # when maze size is provided use the preset parameters, otherwise values will use those specified at the command line
        if usePresetParams:
            self.epsilon = self.getParams("epsilon")
            self.alpha = self.getParams("alpha")
            self.discount = self.getParams("gamma")

        # *** YOUR CODE STARTS HERE ***

        # Flat Q-table: key is (encoded_state, action) -> value (float)
        self.Qtable = util.Counter()

        # Per-episode cache: fixed ordering of the 4 food coordinates for bitmask
        self._food_order = None

        # *** YOUR CODE ENDS HERE ***
    

    def registerInitialState(self, state: GameState):
        """
        Don't modify this method except in the provided area.
        You can modify this method to do any computation you need at the start of each episode
        """

        # *** YOUR CODE STARTS HERE ***

        # Determine a stable order for the 4 food locations for this episode
        food_grid = state.getFood()
        try:
            food_list = food_grid.asList()   # typical Pacman Grid API
        except:
            # fallback: enumerate grid
            food_list = []
            for x in range(food_grid.width):
                for y in range(food_grid.height):
                    if food_grid[x][y]:
                        food_list.append((x, y))
        # Sort for determinism (x then y)
        self._food_order = tuple(sorted(food_list))
        

        # *** YOUR CODE ENDS HERE ***

        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))
    
    
    def getAction(self, state: GameState):
        """
        Don't modify this method!
        Uses epsilon greedy to select an action based on the agents Q table.
        """

        action = self.epsilonGreedyActionSelection(state)
        self.doAction(state, action)
        return action

    def update(self, state: GameState, action: str, nextState: GameState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here using the Q value update equation

        NOTE: You should never call this function,
        it will be called on your behalf
        """

        # *** YOUR CODE HERE ***
        if action is None:
            return

        s_key = self._encode_state(state)
        sp_key = self._encode_state(nextState)

        # Max over next actions (0 if terminal/no legal moves)
        next_actions = [a for a in self.getLegalActions(nextState) if a != Directions.STOP]
        if len(next_actions) == 0:
            max_next = 0.0
        else:
            max_next = max(self.Qtable[(sp_key, a)] for a in next_actions)

        target = reward + self.discount * max_next
        self.Qtable[(s_key, action)] += self.alpha * (target - self.Qtable[(s_key, action)])

    def getParams(self, param_name):
        """
        Add your parameters here 
        """
        params = {
            "gamma": 0.90,
            "epsilon": 0.12,
            "alpha": 0.22
        }
        return params[param_name]


    def epsilonGreedyActionSelection(self, state: GameState):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: When the agent is no longer in training self.epsilon will be set to 0, 
        so calling this method should always return the best action over the Q values
        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        HINT: You might want to use self.getLegalActions(state), 
        but consider whether or not using the STOP action is necessary or beneficial
        """
        
        # *** YOUR CODE HERE ***
        legal = [a for a in self.getLegalActions(state) if a != Directions.STOP]
        if not legal:
            return None

        # Explore
        if util.flipCoin(self.epsilon):
            return random.choice(legal)

        # Exploit: argmax_a Q(s,a) with deterministic tie-break (N,E,S,W)
        s_key = self._encode_state(state)
        pref = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
        # Keep only those in legal while preserving pref order
        ordered = [a for a in pref if a in legal]

        best_a, best_q = None, float('-inf')
        for a in ordered:
            q = self.Qtable[(s_key, a)]
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    ################################ ANY OTHER CODE BELOW HERE ################################

    # ---- Helper utilities (allowed to add here) ----
    def _encode_state(self, state: GameState):
        pac = state.getPacmanPosition()

        # Food bitmask over a fixed order (as you already set in registerInitialState)
        mask = 0
        if self._food_order is not None:
            food_grid = state.getFood()
            for i, (fx, fy) in enumerate(self._food_order):
                present = bool(food_grid[fx][fy]) if food_grid is not None else False
                if present:
                    mask |= (1 << i)

        # NEW: include the (single) moving ghost’s cell (integer coords)
        # (Q2 uses exactly one moving ghost.)
        ghost_pos = state.getGhostPositions()
        if ghost_pos:
            gx, gy = int(ghost_pos[0][0]), int(ghost_pos[0][1])
        else:
            gx, gy = -1, -1  # fallback, shouldn’t happen in Q2

        return (pac, mask, (gx, gy))

    

