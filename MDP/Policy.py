import numpy as np


class Policy:
    def get_action(self, state):
        pass
        # p_state = self.p_mat[dict[state], :]
        # cum_p = np.cumsum(p_state)
        # rnd = np.random.rand()
        # action = actions[np.argmin(rnd < cum_p) + 1]
        # return action

    def set_policy(self, p_mat):
        pass
        # self.p_mat = p_mat

    def __init__(self):
        pass
        # i = 0
        # for state in states:
        #     self.states[state] = i
        #     i += 1
        # j = 0
        # for action in actions:
        #     self.actions[j] = action
        #     j += 1
        # self.p_mat = np.c_[np.ones(len(states), 1), np.zeros(len(states), len(actions) - 1)]
