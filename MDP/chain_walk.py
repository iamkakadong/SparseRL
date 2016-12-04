import numpy as np

class chain_walk(MDP):
    '''
        a = 1: go right
        a = -1: go left
    '''

    length = 0
    rew = list() # Reward vector for faster computation of reward

    def reward(self, s, a):
        return self.rew[s]

    def transit(self, s, a):
        rnd = np.random.rand()
        s_next = s + a * ((rnd < 0.9).real * 2 - 1)
        if s_next > self.length:
            s_next = self.length - 1
        if s_next < 0:
            s_next = 0
        self.set_cur_state(s_next)
        return s_next   # NOTE: This may not be necessary

    def sample(self, policy):
        a = policy.get_action(self.cur_state) # NOTE: policy must have method get_action that returns a value in {-1, 1}
        r = self.reward(self.cur_state, a)
        s_next = self.transit(self.cur_state, a)
        return (a, r, s_next)

    def vf_t(self, policy):
        # TODO: compute true value function
        return 0

    def __init__(self, gamma, length):
        MDP.__init__(self, gamma, set(range(length)))
        self.length = length
        self.rew = [0 for i in range(length)]
        self.rew[0] = 1
        self.rew[-1] = 1

