# Python imports.
import sys, os
import numpy as np

# Other imports.
import tensorflow as tf
from simple_rl.mdp import State
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
import numpy

# TODO:
    # Consider putting different MDP state abstractions into different directories in sa_models.
    # Add sampling to phi().

class NNStateAbstr(StateAbstraction):

    def __init__(self, abstraction_net):
        '''
        Args:
            abstraction_net (str): The name of the model.
        '''
        self.abstraction_net = abstraction_net

    def phi(self, state):
        '''
        Args:
            state (simple_rl.State)

        Returns:
            state (simple_rl.State)
        '''
        pr_z_given_s = list(self.abstraction_net.predict([state]))
        abstr_state_index = np.argmax(pr_z_given_s)

        return State(abstr_state_index)

    def phi_pmf(self, state):
        '''
        Args:
            state (simple_rl.State)

        Returns:
            (list): Contains probabilities. (index is z, value is prob of z given @state).
        '''
        return self.abstraction_net.predict([state])[0]
