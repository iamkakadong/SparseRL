import numpy as np

class Elastic_TD:
    '''
    Variables:
        gamma: discount factor
        epsilon: paramter of our optimization problem
        mu: parameter for admm
        t: step size in proximal gradient
        alpha: elastic-net parameter
        n: number of trajectory samples
        k: number of features
        Phi: current states of trajectory samples, n * k dimension matrix
        Phi_: next states of trajectory samples, n * k dimension matrix
        PI: feature space projector, Phi(Phi.T * Phi)^(-1) * Phi.T, 
            the inverse is computed as pseudo-inverse when singular, n * n matrix
        R: trajectory reward, n dim vector
        A: minimization problem in ||A * theta - b||_nM^2, A = (1/n) * Phi.T * (Phi - gamma * Phi_), k * k matrix
        b: minimization problem in ||A * theta - b||_nM^2, b = (1/n) * Phi.T * R, k dim vector
        M: M norm of ||A * theta - b||_nM^2
        C: minimizatino problem in ||C * theta + d||_2, C = gamma * PI * Phi_ - Phi, n * k matrix
        d: minimizatino problem in ||C * theta + d||_2, d = PI * R, n dim vector

        theta: what we ultimately want to know to compute (V_theta = Phi(s) * theta), k dim vector
        beta: auxiliary variable in ADMM, n dim vetor
        prev_beta: last beta, used for stopping criteria
        z: lagrange multiplier for (C*theta + b - beta)     
    '''
    def map_features(self, states):
        '''Calculate the state representation given the trajectory samples'''

        #define what is the feature in each dimention 1, ... K
        func_list = [lambda x: 1, lambda x: x, lambda x: x*x]
        self.K = len(func_list)

        res = []
        for i in range(len(func_list)):
            f = np.vectorize(func_list[i])
            if i == 0:
                res = f(states)
            else:
                res = np.c_[res, f(states)]
        return res

    def __init__(self, gamma, mu, alpha, eplison, states, next_states, rewards):
        ''' Compute required variables '''
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.eplison = eplison
        self.Phi = np.array(states)
        self.Phi_ = np.array(next_states)
        self.R = np.array(rewards)
        self.PI = np.dot(
                      np.dot(
                          self.Phi,
                          np.linalg.pinv(
                              np.dot(self.Phi.T, self.Phi)
                          )
                      ),
                      self.Phi.T)
        self.n = self.Phi.shape[0]
        self.k = self.Phi.shape[1]
        self.A = (1.0 / self.n) * self.Phi.T.dot(self.Phi - self.gamma * self.Phi_)
        self.M = self.n * np.linalg.pinv(np.dot(self.Phi.T, self.Phi))
        self.b = (1.0 / self.n) * self.Phi.T.dot(self.R)
        self.C = self.gamma * np.dot(self.PI, self.Phi_) - self.Phi
        self.d = np.dot(self.PI, self.R)

        w, v = np.linalg.eig(self.C.T.dot(self.C) + 2*self.mu*(1-self.alpha)*np.eye(self.k))
        self.t = 0.99 / np.max(w)

        '''objs'''
        self.objs = []

        ''' Initialize variables '''
        self.theta = np.zeros(self.k)
        self.beta = np.zeros(self.n)
        self.prev_beta = np.zeros(self.n)
        self.z = np.zeros(self.n)

    def admm_stop(self, itr, ep_prm, ep_dual):
        res_prm = np.linalg.norm( np.dot(self.C, self.theta) + self.d - self.beta )
        res_dual = np.linalg.norm( -1.0 / self.mu  * self.C.T.dot(self.beta - self.prev_beta) )
        return res_prm <= ep_prm and res_dual <= ep_dual

    def soft_thd(self, vec, thred):
        return np.maximum(np.minimum(vec+thred, 0), vec-thred)

    def cal_obj(self):
        res = self.C.dot(self.theta) + self.d
        return res.dot(res)

    def ADMM(self):
        '''run admm solver'''
        i = 0
        while not self.admm_stop(i, 1e-2, 1e-2):
            # update beta
            self.prev_beta = np.copy(self.beta)
            self.beta = np.dot(self.C, self.theta) + self.d - self.mu * self.z
            beta_norm = np.linalg.norm(self.beta)
            if beta_norm > self.eplison:
                self.beta = self.beta * (self.eplison / beta_norm)

            # update theta
            grad = np.dot(self.C.T, np.dot(self.C, self.theta)+self.d-self.beta-self.mu*self.z) + 2*self.mu*(1-self.alpha) * self.theta
            self.theta = self.soft_thd(self.theta - self.t*grad, self.mu * self.alpha * self.t)

            # update z 
            self.z -= 1.0 / self.mu * (np.dot(self.C, self.theta) + self.d - self.beta)

            # calculate objective
            self.objs.append(self.cal_obj())
            i += 1
            # print self.objs[-1]
            if i % 1000 == 0:
                print 'Iteration: ', i, self.objs[-1]
        print i

    def proximal_GD(self):
        pass
