
class MDP:

    gamma = 0.0

    def reward(s, a):
        ''' Reward the agent gets by taking action 'a' at state 's' '''
        pass

    def transit(s, a):
        ''' Transition of state when an agent takes action 'a' at state 's' '''
        pass

    def sample(policy):
        ''' Generate sample according to policy, each sample contains (current_state, next_state, reward) '''
        pass

    def noisy_sample(policy):
        ''' Generate sample according to policy, then add noisy irrelavent features '''
        pass

    def vf_t(policy):
        ''' Return true value function of the given policy '''
        pass


