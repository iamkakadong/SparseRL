import MDP.MDP
from MDP.chain_walk import *
from MDP.pendulum import *
from MDP.pendulum_policy import *
import MDP.Policy
import numpy as np
import td.Elastic as Elastic
import td.elastic_td as elastic_td
from td.lstd import lstd
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

    res = {}
    with open('samples/samples_pendulum.pickle') as handle:
        sets = pickle.load(handle)

    num_sets = 10
    noises = [20, 50, 100, 200, 500, 800]

    # Learning
    for index in range(num_sets):
        for n_noisy in noises:
            state_seq, next_state_seq, reward_seq = sets[(n_noisy, index)]

            state_seq.append(next_state_seq[-1])
            agent = lstd(0.0, 41 + n_noisy, gamma)
            agent.set_start(state_seq[0])
            prev_state = state_seq[0]
            for i in range(len(reward_seq)):
                agent.update_V(prev_state, state_seq[i + 1], reward_seq[i])
                prev_state = state_seq[i + 1]

            theta = agent.get_theta()
        
            mse, truth, pred = env.compute_mse(policy, theta, n_noisy, mc_iter=1000, restart=200)
            res[(n_noisy, index)] = (mse, theta)
            print index, n_noisy, mse

    with open('results/pendulum_res_lstd.pickle', 'wb') as handle:
        pickle.dump(res, handle)