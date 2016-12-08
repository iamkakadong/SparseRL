import numpy as np
from numpy import linalg

class Elastic_TD:
    def __init__(self, n, k, gamma):
        self.n = n
        self.k = k
        self.gamma = gamma
        self.beta = np.zeros(self.k)
        self.tau = 0

    # generaate the feature vector for a sample x
    # e.g phi(x) = [1, x, x^2]
    # x:                n x p
    # output:           n x k
    def phi(self, x):
        I = np.ones(self.n)
        return np.column_stack((I, x, x * x))

    # generate the matrix tilde_Phi, tilde_Phi_prime, tilde_R
    # input: (n + 1) samples X and (n + 1) rewards R
    # output:
    #   tilde_Phi:        n x k (based on x_1 to x_n)
    #   tilde_Phi_pirme:  n x k (based on x_2 to x_(n + 1))
    #   tilde_R:          n x k (based on r_2 to r_(n + 1))
    def calculate_base(self, X, R):
       # tilde_Phi = self.phi(X[0 : self.n])
       # tilde_Phi_prime = self.phi(X[1 : self.n + 1])
       # tilde_R = R[1 : self.n + 1]
        tilde_Phi = X[0 : self.n]
        tilde_Phi_prime = X[1 : self.n + 1]
        tilde_R = R[1 : self.n + 1]
        return tilde_Phi, tilde_Phi_prime, tilde_R

    # generate the model parameters tilde_d, tilde_C, tilde_Pi
    # generate tilde_A, tilde_d, tilde_G for computing the loss
    # input: tilde_Phi, tilde_Phi_prime, tilde_R
    # output:
    #   tilde_d:            n x 1
    #   tilde_C:            n x k
    #   tilde_Pi:           n x n
    #   tilde_A:            k x k
    #   tilde_b:            k x 1
    #   tilde_G:            k x k
    def calculate_param(self, tilde_Phi, tilde_Phi_prime, tilde_R):
        tilde_Phi_T = np.transpose(tilde_Phi)

        # for admm algorithm
        tilde_Pi = np.dot(
                      np.dot(
                          tilde_Phi,
                          linalg.pinv(
                              np.dot(tilde_Phi_T, tilde_Phi)
                          )
                      ),
                      tilde_Phi_T)
        tilde_d = np.dot(tilde_Pi, tilde_R)
        tilde_C = self.gamma * np.dot(tilde_Pi, tilde_Phi_prime) - tilde_Phi

        # for computing loss
        tilde_A = 1.0 / self.n * (np.dot(tilde_Phi_T, tilde_Phi)
                        - self.gamma * np.dot(tilde_Phi_T, tilde_Phi_prime))
        tilde_b = 1.0 / self.n * np.dot(tilde_Phi_T, tilde_R)
        tilde_G = self.n * linalg.pinv(np.dot(tilde_Phi_T, tilde_Phi))
        return tilde_d, tilde_C, tilde_Pi, tilde_A, tilde_b, tilde_G

    ########## proximal gradient descent step in ADMM  ##########
    # f denotes the smooth part
    # operator for proximal gradient
    def prox(self, eta, x):
        return np.maximum(
                np.minimum(x + eta, 0),
                x - eta
               )

    # calculate the gradient for f
    def grad(self, tilde_C, tilde_d, beta, alpha, mu, delta, v):
        return np.dot(
                np.transpose(tilde_C),
                np.dot(
                    tilde_C,
                    beta
                )
                + tilde_d - alpha - mu * v
            ) + 2 * mu * (1 - delta) * beta

    # compute the step size tau
    def compute_tau(self, tilde_C, mu, delta):
        eigen_val, _  = linalg.eig(np.dot(np.transpose(tilde_C), tilde_C)
                        + mu * (1 - delta) * np.identity(self.k))
        self.tau = 0.1 / np.max(eigen_val)

    ########## projection part ##########
    # solution for projection problem
    def solve_proj(self, c, epsilon):
        norm_c = linalg.norm(c)
        if norm_c <= epsilon:
            alpha = c
        else:
            alpha = epsilon * c / norm_c
        return alpha

    ########## main algorrithm in ElasticRL (ADMM) ##########
    # input:
    #   tilde_C:            n x k
    #   tilde_d:            n x 1
    #   mu:                 parameter for augmented Lagrangian
    #   tau:                parameter for proximal gradient descent
    #   epsilon:            parameter for projection problem
    #   delta:              parameter for the l1-norm and l2-norm
    # output:
    #   beta:               learned coefficient
    def elastic_td(self, tilde_C, tilde_d, tilde_A, tilde_b, tilde_G, mu, epsilon, delta, epoch):
        # initialize parameters
        v = np.zeros(self.n)
        alpha = np.zeros(self.n)
        self.compute_tau(tilde_C, mu, delta)

        # admm updates
        for j in range(epoch):
            c = tilde_d + np.dot(tilde_C, self.beta) - mu * v
            alpha = self.solve_proj(tilde_d + np.dot(tilde_C, self.beta) - mu * v, epsilon)
            self.beta = self.prox(self.tau * mu * delta, self.beta - self.tau * self.grad(tilde_C, tilde_d, self.beta, alpha, mu, delta, v))
            v = v - 1 / mu * (tilde_d + np.dot(tilde_C, self.beta) - alpha)
            print(self.compute_loss(tilde_A, tilde_b, tilde_G))
        return self.beta

    # compute the MSPBE
    def compute_loss(self, tilde_A, tilde_b, tilde_G):
        v = np.dot(tilde_A, self.beta) - tilde_b
        return np.dot(np.dot(np.transpose(v), self.n * tilde_G), v)

    def run(self, mu, epsilon, delta, epoch, X, R):
        tilde_Phi, tilde_Phi_prime, tilde_R = self.calculate_base(X, R)
        tilde_d, tilde_C, _, tilde_A, tilde_b, tilde_G = self.calculate_param(tilde_Phi, tilde_Phi_prime, tilde_R)
        self.beta = self.elastic_td(tilde_C, tilde_d, tilde_A, tilde_b, tilde_G, mu, epsilon, delta, epoch)
        return self.beta

if __name__=='__main__':
    mu = 0.01
    tau = 0.05
    epsilon = 0.1
    gamma = 0.9
    n = 2
    k = 3
    alg = Sparse_TD(n, k, gamma)
    X = np.array([1, 2, 3])
    R = np.array([2, 3, 4])
    alg.run(mu, tau, epsilon, X, R)
