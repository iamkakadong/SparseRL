import MDP.MDP
from MDP.chain_walk import *
from MDP.pendulum import *
from MDP.pendulum_policy import *
import MDP.Policy
import numpy as np
import td.Elastic as Elastic
import td.elastic_td as elastic_td
from td.lstd import lstd

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

    n_noisy = 800
    n_samples = 1000

    # Generate a sequence of 1000 noisy samples with 20 irrelavent features from the environment
    state_seq = list()
    next_state_seq = list()
    action_seq = list()
    reward_seq = list()
    state = env.get_noisy_state(n_noisy)
    for i in range(n_samples):
        # Each sample is a tuple (action, reward, next state)
        state_seq.append(state)
        sample = env.noisy_sample_corr(policy, n_noisy)
        action_seq.append(sample[0])
        reward_seq.append(sample[1])
        next_state_seq.append(sample[2])
        state = sample[2]

    # Learning
    agent = lstd(0.0, 41 + n_noisy, gamma)
    agent.set_start(state_seq[0])
    prev_state = state_seq[0]
    for i in range(len(reward_seq)):
        agent.update_V(prev_state, state_seq[i + 1], reward_seq[i])
        prev_state = state_seq[i + 1]

    # Examine result
    theta = agent.get_theta()
    # print theta

    mse, truth, pred = env.compute_mse(policy, theta, n_noisy, mc_iter=1000, restart=200)
    print mse