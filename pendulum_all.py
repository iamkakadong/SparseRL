import MDP.MDP
from MDP.pendulum import *
from MDP.pendulum_policy import pendulum_policy
import MDP.Policy
import numpy as np
import td.Elastic_fast as Elastic_fast
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    gamma = 0.95
    dim = 20
    length = 1
    mass = 1
    sigma = 0.01
    dt = 0.01
    penalty = 0.01
    action_penalty = 0.0

    policy_noise = 0.1

    # Define environment and policy
    env = pendulum(dim, length, mass, sigma, dt, gamma, penalty, action_penalty)
    policy = pendulum_policy(dim * 2, dim, policy_noise)

    # Compute optimal policy via dynamic programming
    theta_p, _, _ = env.get_opt_policy()

    # Set policy to optimal policy, i.e. move left if state < 10, move right if state >= 10 (state index start with 0)
    policy.set_policy(theta_p, policy_noise)

    # Get true value function for the policy
    vf = env.get_vf(policy)

    # Set current state of environment to 0
    env.reset_state()

    #set parameters for solver
    epsilon = 0.01
    mu = 1
    alpha = 1.0
    eta = 0.9

    res = {}
    with open('samples/samples_pendulum.pickle') as handle:
        sets = pickle.load(handle)

    num_sets = 10
    noises = [800]
    for index in range(num_sets):
        for n_noisy in noises:
            print index, n_noisy
            states, next_states, rewards = sets[(n_noisy, index)]
            solver = Elastic_fast.Elastic_TD(gamma, mu, alpha, eta, epsilon, states, next_states, rewards)
            solver.ADMM()

            mse, truth, pred = env.compute_mse(policy, solver.theta, n_noisy, mc_iter=1000, restart=200)

            res[(n_noisy, index)] = (mse, solver.theta)
            print mse

    with open('results/res_a1.pickle', 'wb') as handle:
        pickle.dump(res, handle)