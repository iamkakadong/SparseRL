import MDP.MDP
from MDP.chain_walk import *
import MDP.Policy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import td.lstd as lstd
import matplotlib.pyplot as plt

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

    res = {}
    with open('samples/samples.pickle') as handle:
        sets = pickle.load(handle)

    # running lstd
    num_sets = 10
    noises = [20, 50, 100, 200, 500, 800]
    for index in range(num_sets):
        for n_noisy in noises:
            state_seq, next_state_seq, reward_seq = sets[(n_noisy, index)]
            agent = lstd.lstd(0.0, 3 + n_noisy, gamma)

            state_seq.append(next_state_seq[-1])
            agent.set_start(state_seq[0])
            prev_state = state_seq[0]
            for i in range(len(reward_seq)):
                if i == 500:
                    agent.set_start(state_seq[i])
                    prev_state = state_seq[i]
                else:
                    agent.update_V(prev_state, state_seq[i + 1], reward_seq[i])
                    prev_state = state_seq[i + 1]

            state_seq.pop()
            theta = agent.get_theta()

            mse, truth, pred = env.compute_mse(policy, theta, n_noisy, mc_iter=1000, restart=200)

            res[(n_noisy, index)] = (mse, theta)
            print index, n_noisy, mse

    with open('results/res_lstd.pickle', 'wb') as handle:
        pickle.dump(res, handle)