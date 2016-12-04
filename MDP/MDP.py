import numpy as np

class MDP:

    gamma = 0.0 # Discount factor
    states = set() # Not used
    actions = set() # Not used
    cur_state = None

    def reward(self, s, a):
        ''' Reward the agent gets by taking action 'a' at state 's' '''
        pass

    def transit(self, s, a):
        ''' Transition of state when an agent takes action 'a' at state 's' '''
        pass

    def sample(self, policy):
        ''' Generate sample according to policy, each sample contains (current_state, next_state, reward) '''
        pass

    def noisy_sample(self, policy, n_irrel):
        ''' Generate sample according to policy, then add noisy irrelavent features '''
        sample = self.sample(policy)
        s_next = sample[2]
        f_irrel = np.random.randn(n_irrel)
        sample[2] = np.r_[s_next, f_irrel]
        return sample

    def vf_t(self, policy):
        ''' Return true value function of the given policy '''
        pass

    def to_features(self, state):
        ''' Convert actual state to a feature representation '''
        pass

    def set_cur_state(self, cur_state):
        self.cur_state = cur_state

    def set_actions(self, actions):
        self.actions = actions

    def get_actions(self):
        return self.actions

    def __init__(self, gamma):
        self.gamma = gamma

