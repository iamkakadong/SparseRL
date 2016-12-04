from Policy import Policy
import numpy as np


class chain_walk_policy(Policy):

    length = 0
    p_mat = 0.0

    def get_action(self, state):
        p_state = self.p_mat[state, :]
        # cum_p = np.cumsum(p_state)
        # rnd = np.random.rand()
        # action = (np.argmin(rnd < cum_p)) * 2 - 1
        action = np.flatnonzero(np.random.multinomial(1, p_state))[0] * 2 - 1
        return action

    def set_policy(self, p_mat):
        self.p_mat = p_mat

    def get_p(self):
        return self.p_mat

    def __init__(self, length):
        Policy.__init__(self)
        self.length = length
        self.p_mat = np.zeros([length, 2])
