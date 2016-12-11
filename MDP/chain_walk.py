import numpy as np

from MDP import MDP
from chain_walk_policy import chain_walk_policy


class chain_walk(MDP):
    '''
        a = 1: go right
        a = -1: go left
    '''

    length = 0
    rew = list()  # Reward vector for faster computation of reward

    def reward(self, s, a):
        '''
            Receive +1 reward if the state is the agent is at the first or the last state
            Receive 0 reward if otherwise
        '''
        return self.rew[s]

    def transit(self, s, a):
        '''
            With probability 0.9, the agent transit to the next state consistent with the action
            With probability 0.1, the agent transit to the next state opposite to the action
            If the agent is at a boundary state, it remains at the current state if the next state is outside of the state space
        '''
        rnd = np.random.rand()
        s_next = s + a * ((rnd < 0.9).real * 2 - 1)
        if s_next >= self.length:
            s_next = self.length - 1
        if s_next < 0:
            s_next = 0
        self.set_cur_state(s_next)
        return s_next  # NOTE: This may not be necessary

    # def sample(self, policy):
    #     '''
    #         Returns a sample following the policy starting from the current state.
    #         Sample is a tuple:
    #             (action, reward, next_state)
    #     '''
    #     a = policy.get_action(self.cur_state)  # NOTE: policy must have method get_action that returns a value in {-1, 1}
    #     r = self.reward(self.cur_state, a)
    #     s_next = self.transit(self.cur_state, a)
    #     return (a, r, self.to_features(s_next))

    def compute_mse(self, policy, theta, n_irrel, mc_iter=100, restart=1000):
        '''
            Compute MSE = ||V_pi - V_theta||_D^2 of a policy and value          function approximator theta.
            Use Monte-Carlo method to approximate stationary distribution of    the system.
            mc_iter: total monte-carlo simulations initiated
            restart: the length of each monte-carlo chain
        '''
        vf = self.get_vf(policy)
        truth = list()
        pred = list()
        i = 0
        for i in range(mc_iter):
            self.reset_state()
            for j in range(restart):
                s = self.cur_state
                truth.append(vf[s])
                pred.append(np.inner(theta, np.r_[self.to_features(s), np.random.randn(n_irrel)]))
                self.transit(s, policy.get_action(s))
        mse = np.mean(map(lambda x, y: np.linalg.norm(x - y) ** 2, truth, pred))
        return mse, truth, pred

    def get_vf(self, policy):
        '''
            Compute true value function by solving the system of linear equations defined by the Bellman equation
        '''
        #assert (isinstance(policy, chain_walk_policy))
        p_mat = policy.get_p()  # chain_walk_policy has function get_p which returns a probability matrix p(a|s)
        A = np.zeros([self.length, self.length])
        for i in range(self.length):
            A[i, np.max([i - 1, 0])] = self.gamma * (p_mat[i, 0] * 0.9 + p_mat[i, 1] * 0.1)
            A[i, np.min([i + 1, self.length - 1])] = self.gamma * (p_mat[i, 0] * 0.1 + p_mat[i, 1] * 0.9)
        b = np.zeros([self.length, 1])
        b[0] = 1
        b[-1] = 1
        vf = np.linalg.solve(np.eye(self.length) - A, b)
        return vf

    def get_stationary(self, policy):
        p = np.zeros([self.length, self.length])
        p_mat = policy.get_p()
        for i in range(20):
            p[i, max(i - 1, 0)] = 0.9 * p_mat[i, 0] + 0.1 * p_mat[i, 1]
            p[i, min(i + 1, 19)] = 0.1 * p_mat[i, 0] + 0.9 * p_mat[i, 1]
        p = p.T - np.eye(self.length)
        p[-1,:] = 1
        b = [0 for i in range(20)]
        b[-1] = 1
        s_dist = np.linalg.solve(p, b)
        if any(s_dist < 0):
            print 'Error! Stationary distribution has negative component'
            return None
        return s_dist

    def to_features(self, state):
        f_s = [1, state, state ** 2]
        return f_s

    def __init__(self, gamma, length):
        MDP.__init__(self, gamma)
        self.set_actions({-1, 1})
        self.length = length
        self.rew = [0 for i in range(length)]
        self.rew[0] = 1
        self.rew[-1] = 1
        self.DEFAULT_STATE = 10
