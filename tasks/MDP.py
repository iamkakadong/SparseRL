
class MDP:

    gamma = 0.0 # Discount factor
    states = set()
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

    def noisy_sample(self, policy):
        ''' Generate sample according to policy, then add noisy irrelavent features '''
        pass

    def vf_t(self, policy):
        ''' Return true value function of the given policy '''
        pass

    def __init__(self, gamma):
        self.gamma = gamma

