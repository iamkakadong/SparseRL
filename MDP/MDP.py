import numpy as np

class MDP:

    gamma = 0.0 # Discount factor
    states = set() # Not used
    actions = set() # Not used
    cur_state = None
    noisy_state = None

    def reward(self, s, a):
        ''' Reward the agent gets by taking action 'a' at state 's' '''
        pass

    def transit(self, s, a):
        ''' Transition of state when an agent takes action 'a' at state 's' '''
        pass

    def sample(self, policy):
        '''
            Returns a sample following the policy starting from the current state.
            Sample is a tuple:
                (action, reward, next_state)
        '''
        a = policy.get_action(self.cur_state)  # NOTE: policy must have method get_action that returns a value in the actions space
        r = self.reward(self.cur_state, a)
        s_next = self.transit(self.cur_state, a)
        return (a, r, self.to_features(s_next))

    def noisy_sample(self, policy, n_irrel):
        ''' Generate sample according to policy, then add noisy irrelavent features '''
        sample = self.sample(policy)
        s_next = sample[2]
        f_irrel = np.random.randn(n_irrel)
        sample = (sample[0], sample[1], np.r_[s_next, f_irrel])
        return sample

    def get_vf(self, policy):
        ''' Return true value function of the given policy '''
        pass

    def get_noisy_state(self, n_irrel):
        return np.r_[self.to_features(self.cur_state), np.random.randn(n_irrel)]

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

