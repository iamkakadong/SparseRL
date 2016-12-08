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
        beta: auxiliary variable in ADMM, n
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

    def __init__(self, gamma, mu, alpha, eplison, states, rewards):
        ''' Compute required variables '''
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.eplison = eplison
        self.Phi = np.array(states[0:-1])
        self.Phi_ = np.array(states[1:])
        self.n = self.Phi.shape[0]
        self.k = self.Phi.shape[1]
        self.R = np.array(rewards)
        self.PI = np.dot(self.Phi.dot(np.linalg.pinv(np.dot(self.Phi.T, self.Phi))), self.Phi.T)
        self.A = (1.0 / self.n) * self.Phi.T.dot(self.Phi - self.gamma * self.Phi_)
        self.M = self.n * np.linalg.pinv(np.dot(self.Phi.T, self.Phi))
        self.b = (1.0 / self.n) * self.Phi.T.dot(self.R)
        self.C = self.gamma * self.PI.dot(self.Phi_) - self.Phi
        self.d = self.PI.dot(self.R)

        w, v = np.linalg.eig(self.C.T.dot(self.C) + 2*self.mu*(1-self.alpha)*np.eye(self.k))
        self.t = 1.0 / np.max(w)

        '''objs'''
        self.objs = []

        ''' Initialize variables '''
        self.theta = np.zeros(self.k)
        self.beta = np.zeros(self.n)
        self.z = np.zeros(self.n)

    def admm_stop(self, itr):
        return itr > 200

    def soft_thd(self, vec, thred):
        return np.maximum(np.minimum(vec+thred, 0), vec-thred)

    def cal_obj(self):
        res = self.C.dot(self.theta) + self.d
        return res.dot(res)

    def ADMM(self):
        '''run admm solver'''
        i = 0
        while not self.admm_stop(i):
            # update beta
            self.beta = np.dot(self.C, self.theta) + self.d - (self.mu * self.z)
            beta_norm = np.linalg.norm(self.beta)
            if beta_norm > self.eplison:
                self.beta = self.beta * (self.eplison / beta_norm)

            # update theta
            grad = np.dot(self.C.T, self.C.dot(self.theta) + self.d - self.beta - self.mu * self.z) + 2*self.mu*(1-self.alpha) * self.theta
            print self.t
            self.theta = self.soft_thd(self.theta - self.t*grad, self.mu * self.alpha * self.t)

            # update z 
            print self.theta
            print self.C.dot(self.theta)
            print (1.0 / self.mu) * (self.C.dot(self.theta) + self.d - self.beta)
            self.z -= (1.0 / self.mu) * (self.C.dot(self.theta) + self.d - self.beta)

            # calculate objective
            self.objs.append(self.cal_obj())
            i += 1

    def proximal_GD(self):
        pass