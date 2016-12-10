from Policy import Policy
import numpy as np


class pendulum_policy(Policy):

    theta = 0.
    dim_S = 0
    dim_A = 0
    noise = 0.

    def get_action(self, state):
        m = np.array(np.dot(self.theta, state)).flatten()
        noise = np.sqrt((np.ones((self.dim_A)) * self.noise)[None, :]) * np.random.randn(self.dim_A)
        return (m + noise).flatten()

    def set_policy(self, theta, noise):
        self.theta = theta
        self.noise = noise

    def __init__(self, dim_S, dim_A, noise):
        self.dim_S = dim_S
        self.dim_A = dim_A
        self.theta = np.zeros([dim_A, dim_S])
        self.noise = noise
