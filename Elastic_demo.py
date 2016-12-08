import MDP.MDP
from MDP.chain_walk import *
import MDP.Policy
import numpy as np
from td.Elastic_TD import *

if __name__ == "__main__":
    gamma = 0.9
    length = 20

    #set parameters for solver
    epsilon = 0.01
    mu = 10
    alpha = 1

    # Define environment and policy
    env = chain_walk(gamma, length)
    policy = chain_walk_policy(length)

    # Set policy to optimal policy, i.e. move left if state < 10, move right if state >= 10 (state index start with 0)
    p_mat = np.zeros([20, 2])
    p_mat[0:10, 0] = 1
    p_mat[10::, 1] = 1
    policy.set_policy(p_mat)

    # Get true value function for the policy
    vf = env.get_vf(policy)

    # Set current state of environment to 0
    env.set_cur_state(9)

    # Generate a sequence of 1000 noisy samples with 20 irrelavent features from the environment
    n_noisy = 20
    n_samples = 10
    state_seq = list()
    action_seq = list()
    reward_seq = list()
    state_seq.append(env.get_noisy_state(n_noisy))
    for i in range(n_samples):
        # Each sample is a tuple (action, reward, next state)
        sample = env.noisy_sample(policy, n_noisy)
        action_seq.append(sample[0])
        reward_seq.append(sample[1])
        state_seq.append(sample[2])

    solver = Elastic_TD(gamma, mu, alpha, epsilon, state_seq, reward_seq)
    solver.ADMM()
    print solver.theta
    print solver.objs