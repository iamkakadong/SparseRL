import numpy as np

from MDP import MDP

class pendulum(MDP):

    dim = 0
    l = 0
    g = 9.81
    m = 0.0
    sigma = 0.0
    dt = 0.0

    A = 0.0
    B = 0.0
    M = 0.0
    U = 0.0
    R = 0.0
    Q = 0.0

    cur_state = None # A 2 * dim vector; First half are angles of the joints, second half are angular velocities of the joints

    def reward(self, s, a):
        return - np.linalg.norm(s[:self.dim]) ** 2

    def transit(self, s, a):
        z = np.random.randn(self.dim * 2) * self.sigma
        s_next = np.dot(self.A, s) + np.dot(self.B, a) + z
        self.cur_state = s_next
        return s_next

    def get_opt_policy(self, n_iter=100000, eps=1e-14):
        # From Christoph Dann's code on Github: tdlearn -> dynamic_prog.py -> solve_LQR
        ''' Solve for the optimal policy of this problem via Dynamic Programming'''

        P = np.matrix(np.zeros([self.dim * 2, self.dim * 2]))
        R = np.matrix(self.R)
        b = 0.0
        theta = np.matrix(np.zeros([self.dim, self.dim * 2]))
        A = self.A
        B = self.B
        gamma = self.gamma
        for i in xrange(n_iter):
            theta_n = - gamma * np.linalg.pinv(R + gamma * B.T * P * B) * B.T * P * A
            P_n, b_n = self.bellman_operator(P, b, theta, gamma)
            if np.linalg.norm(P - P_n) < eps and np.abs(b - b_n) < eps and np.linalg.norm(theta - theta_n) < eps:
                print "Converged estimating V after ", i, "iterations"
                break
            P = P_n
            b = b_n
            theta = theta_n
        return np.asarray(theta), P, b

    def bellman_operator(self, P, b, theta, gamma, noise=0.0):
        # From Christoph Dann's code on Github: tdlearn -> dynamic_prog.py -> solve_LQR
        ''' Bellman operator for the behavioral policy defined with P, b, and theta '''

        Sigma = np.matrix(np.diag(np.ones(self.dim * 2) * self.sigma))
        theta = np.matrix(theta)
        if noise == 0.:
            noise = np.zeros((theta.shape[0]))
        else:
            noise = np.zeros((theta.shape[0])) + noise
        S = self.A + self.B * theta
        C = self.Q + theta.T * self.R * theta

        P_n = C + self.gamma * (S.T * np.matrix(P) * S)
        b_n = gamma * (b + np.trace(np.matrix(P) * np.matrix(Sigma))) + \
                np.trace((self.R + self.gamma * self.B.T * np.matrix(P) * self.B) * np.matrix(np.diag(noise)))
        return P_n, b_n

    def get_vf(self, policy, n_iter=100000, eps=1e-14):
        # From Christoph Dann's code on Github: tdlearn -> dynamic_prog.py -> estimate_V_LQR
        ''' Evaluate the value function for the policy.
            To evaluate the value of a state:
            V = s^T P s
        '''

        T = lambda x, y: self.bellman_operator(x, y, policy.theta, self.gamma, policy.noise)
        P = np.matrix(np.zeros([policy.dim_S, policy.dim_S]))
        b = 0.
        for i in xrange(n_iter):
            P_n, b_n = T(P, b)
            if np.linalg.norm(P - P_n) < eps and np.abs(b - b_n) < eps:
                print "Converged estimating V after ", i, "iterations"
                break
            P = P_n
            b = b_n
        return np.array(P), b

    def get_value(self, s, P):
        return np.dot(s.T, np.dot(P, s))

    def compute_mse(self, policy, theta, n_irrel, mc_iter=100, restart=1000):
        '''
            Compute MSE = ||V_pi - V_theta||_D^2 of a policy and value function approximator theta.
            Use Monte-Carlo method to approximate stationary distribution of the system.
            mc_iter: total monte-carlo simulations initiated
            restart: the length of each monte-carlo chain
        '''
        P, b = self.get_vf(policy)
        truth = list()
        pred = list()
        i = 0
        for i in range(mc_iter):
            self.reset_state()
            for j in range(restart):
                s = self.cur_state
                truth.append(self.get_value(s, P))
                pred.append(np.inner(theta, np.r_[self.to_features(s), np.random.randn(n_irrel)]))
                self.transit(s, policy.get_action(s))
        mse = np.mean(map(lambda x, y: np.linalg.norm(x - y) ** 2, truth, pred))
        return mse, truth, pred

    def to_features(self, state):
        return np.r_[np.array(state) ** 2, 1]

    def __init__(self, dim, l, m, sigma=0.01, dt=0.01, gamma=0.95, penalty=0.01, action_penalty=0.0):
        MDP.__init__(self, gamma)
        self.dim = dim
        self.l = l
        self.m = m
        self.sigma = sigma
        self.dt = dt

        A = np.eye(2 * dim)
        A[:dim, dim:] += dt * np.eye(dim)
        ms = np.array(range(1, dim + 1)[::-1]) * m
        self.M = l ** 2 * np.minimum(ms[:, None], ms)
        Minv = np.linalg.pinv(self.M)
        Upp = - self.g * self.l * ms
        A[dim:, :dim] -= dt * Minv * Upp[None, :]
        self.A = A

        B = np.zeros([2 * dim, dim])
        B[dim:, :] += dt * Minv
        self.B = B

        self.R = np.eye(dim) * action_penalty

        Q = np.zeros([2 * dim, 2 * dim])
        Q[:dim, :dim] += np.eye(dim) * penalty
        self.Q = Q

        self.cur_state = np.zeros(2 * dim)
        self.DEFAULT_STATE = np.copy(self.cur_state)
