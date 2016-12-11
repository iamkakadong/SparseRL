import numpy as np

class lstd:

    lam = 0. # Used for eligibility traces. For ordinary LSTD, set to 0.0. Larger lambda makes the learning algorithm more like Monte-Carlo methods. lambda in [0, 1]
    dim = 0
    gamma = 0.

    t = 0
    Z = None
    b = None
    C = None
    theta = None

    def update_V(self, s0, s1, r):
        '''
            Update parameters to keep track of seen samples
            s0: current state
            s1: next state
            r: reward
        '''
        gamma_t = 1. / (1 + self.t)
        self.Z = self.lam * self.gamma * self.Z + s0
        self.b = (1 - gamma_t) * self.b + gamma_t * self.Z * r
        self.C = (1 - gamma_t) * self.C + gamma_t * np.outer(self.Z, self.gamma * s1 - s0)
        self.t += 1

    def get_theta(self):
        '''
            Compute and return theta per request
        '''
        theta = - np.dot(np.linalg.pinv(self.C), self.b)
        self.theta = theta
        return theta

    def set_start(self, state):
        '''
            Set start state for a new sequence of observations starting at state
        '''
        self.Z = state

    def reset(self):
        '''
            Reset the parameters of the agent
        '''
        dim = self.dim
        self.theta = np.zeros(dim)
        self.Z = np.zeros(dim)
        self.b = np.zeros(dim)
        self.C = np.zeros((dim, dim))
        self.t = 0

    def __init__(self, lam, dim, gamma):
        '''
            Initialize a LSTD agent
            lam: the lambda parameter for LSTD(lambda)
            dim: the dimension of state
            gamma: the time-discount factor
        '''
        self.lam = lam
        self.dim = dim
        self.gamma = gamma

        self.reset()

