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

    # Get true value function for the policy
    vf = env.get_vf(policy)

    # Generate sequence of noisy samples for each state
    num_sets = 10
    noises = [20, 50, 100, 200, 500, 800]
    sample_set = {}
    for index in range(num_sets):
        for n_noisy in noises:
            print index, n_noisy
            states = []
            next_states = []
            actions = []
            rewards = []

            # n_noisy = 800
            n_samples = 1000
            for i in [9,10]:
                # set current state
                env.set_cur_state(i)
                state_seq = []
                action_seq = []
                reward_seq = []
                state_seq.append(env.get_noisy_state(n_noisy))

                #get trajectory samples
                for j in range(n_samples / 2):
                    sample = env.noisy_sample_corr(policy, n_noisy)
                    action_seq.append(sample[0])
                    reward_seq.append(sample[1])
                    state_seq.append(sample[2])

                # Add trajectory to the list
                states += state_seq[:-1]
                next_states += state_seq[1:]
                actions += action_seq
                rewards += reward_seq
            sample_set[(n_noisy, index)] = (states, next_states, rewards)

    with open('samples/samples.pickle', 'wb') as handle:
        pickle.dump(sample_set, handle)