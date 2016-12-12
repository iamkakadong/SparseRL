import MDP.MDP
from MDP.chain_walk import *
import MDP.Policy
import numpy as np
import td.Elastic as Elastic
import td.fast_elastic_td as elastic_td
import td.Elastic_fast as Elastic_fast
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    gamma = 0.9
    length = 20

    # Define environment and policy
    env = chain_walk(gamma, length)
    policy = chain_walk_policy(length)

    # Set policy to optimal policy, i.e. move left if state < 10, move right if state >= 10 (state index start with 0)
    p_mat = np.zeros([20, 2])
    p_mat[0:10, 0] = 1
    p_mat[10::, 1] = 1
    policy.set_policy(p_mat)

    #set parameters for solver
    epsilon = 0.01
    mu = 10
    alpha = 1.0
    eta = 0.5

    res = {}
    with open('samples/samples.pickle') as handle:
        sets = pickle.load(handle)

    num_sets = 10
    noises = [20, 50, 100, 200, 500, 800]
    for index in range(num_sets):
        for n_noisy in noises:
            print index, n_noisy
            states, next_states, rewards = sets[(n_noisy, index)]
            solver = Elastic_fast.Elastic_TD(gamma, mu, alpha, eta, epsilon, states, next_states, rewards)
            solver.ADMM()

            mse, truth, pred = env.compute_mse(policy, solver.theta, n_noisy, mc_iter=1000, restart=200)

            res[(n_noisy, index)] = (mse, solver.theta)

    with open('results/res_lstd.pickle', 'wb') as handle:
        pickle.dump(res, handle)