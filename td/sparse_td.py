import numpy as np
from numpy import linalg

# generaate the feature vector for a sample x
# e.g phi(x) = [1, x, x^2]
# x:                n x p
# output:           n x k
def phi(x):
    n = x.shape[0]
    I = np.ones(n)
    return np.column_stack((I, x, x * x))

# generate the matrix tilde_Phi, tilde_Phi_prime, tilde_R
# input: (n + 1) samples X and (n + 1) rewards R
# output:
#   tilde_Phi:        n x k (based on x_1 to x_n)
#   tilde_Phi_pirme:  n x k (based on x_2 to x_(n + 1))
#   tilde_R:          n x k (based on r_2 to r_(n + 1))
def calculate_base(X, R):
    n = X.shape[0] - 1
    tilde_Phi = phi(X[1 : n])
    tilde_Phi_prime = phi(X[2 : n + 1])
    tilde_R = R[2 : n + 1]
    return tilde_Phi, tilde_Phi_prime, tilde_R

# generate the model parameters tilde_d, tilde_C, tilde_Pi
# input: gamma, tilde_Phi, tilde_Phi_prime, tilde_R
# output:
#   tilde_d:            n x 1
#   tilde_C:            n x k
#   tilde_Pi:           n x n
def calculate_param(gamma, tilde_Phi, tilde_Phi_prime, tilde_R):
    tilde_Phi_T = np.transpose(tilde_Phi)
    tilde_Pi = np.dot(
                  np.dot(
                      tilde_Phi,
                      linalg.pinv(
                          np.dot(tilde_Phi_T, tilde_Phi)
                      )
                  ),
                  tilde_Phi_T)
    tilde_d = np.dot(tilde_Pi, tilde_R)
    tilde_C = gamma * np.dot(tilde_Pi, tilde_Phi_prime) - tilde_Phi
    return tilde_d, tilde_C, tilde_Pi

########## proximal gradient descent step in ADMM  ##########
# f denotes the smooth part
# operator for proximal gradient
def prox(eta, x):
    return np.maximum(
            np.minimum(x + eta, 0),
            x - eta
           )

# calculate the gradient for f
def grad(tilde_C, tilde_d, beta, alpha, mu, v):
    return np.dot(
            np.transpose(tilde_C),
            np.dot(
                tilde_C,
                beta
            )
            + tilde_d - alpha - mu * v
        )

########## projection part ##########
# solution for projection problem
def solve_proj(c, epsilon):
    norm_c = linalg.norm(c)
    if norm_c <= epsilon:
        alpha = c
    else:
        alpha = epsilon * c / norm_c
    return alpha

########## main algorrithm in SparseRL (ADMM) ##########
# input:
#   tilde_C:            n x k
#   tilde_d:            n x 1
#   mu:                 parameter for augmented Lagrangian
#   tau:                parameter for proximal gradient descent
#   epsilon:            paramter for projection problem
# output:
#   beta:               learned coefficient
def sparse_td(tilde_C, tilde_d, mu, tau, epsilon):
    k = tilde_C.shape[1]
    n = tilde_C.shape[0]

    # initialize parameters
    beta = np.zeros(k)
    v = np.zeros(n)
    alpah = np.zeros(n)

    # admm updates
    for j in range(100):
        c = tilde_d + np.dot(tilde_C, beta) - mu * v
        alpha = solve_proj(tilde_d + np.dot(tilde_C, beta) - mu * v, epsilon)
        beta = prox(tau * mu, beta - tau * grad(tilde_C, tilde_d, beta, alpah, mu, v))
        v = v - 1 / mu * (tilde_d + np.dot(tilde_C, beta) - alpha)

    return beta

def run(mu, tau, epsilon, gamma, X, R):
    tilde_Phi, tilde_Phi_prime, tilde_R = calculate_base(X, R)
    tilde_d, tilde_C, _ = calculate_param(gamma, tilde_Phi, tilde_Phi_prime, tilde_R)
    beta = sparse_td(tilde_C, tilde_d, mu, tau, epsilon)

mu = 0.5
tau = 0
epsilon = 0
gamma = 1
X = np.array([1, 2, 3])
R = np.array([2, 3, 4])
run(mu, tau, epsilon, gamma, X, R)
