import MDP.chain_walk as chain_walk
import MDP.chain_walk_policy as chain_walk_policy
import numpy as np
import td.elastic_td as sparse_td

if __name__ == '__main__':

    gamma = 0.9
    length = 20

    # Define environment and policy
    env = chain_walk.chain_walk(gamma, length)
    policy = chain_walk_policy.chain_walk_policy(length)

    # Set policy to optimal policy, i.e. move left if state < 10, move right if      state >= 10 (state index start with 0)
    p_mat = np.zeros([20, 2])
    p_mat[0:10, 0] = 1
    p_mat[10::, 1] = 1
    policy.set_policy(p_mat)

    # Get true value function for the policy
    vf = env.get_vf(policy)

    # Set current state of environment to 0
    env.set_cur_state(9)

    # Generate a sequence of 1000 noisy samples with 20 irrelavent features from     the environment
    n_noisy = 3
    n_samples = 1000
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

    mu = 10
    epsilon = 0.01
    delta = 0.5
    alg = sparse_td.Sparse_TD(n_samples - 1, n_noisy + 3, gamma)
    beta = alg.run(mu, epsilon, delta, np.array(state_seq), np.array(reward_seq))

    print(alg.tau)
    print(beta)
