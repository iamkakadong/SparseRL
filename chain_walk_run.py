import MDP.chain_walk as chain_walk
import MDP.chain_walk_policy as chain_walk_policy
import numpy as np
import td.elastic_td as elastic_td
import td.sparse_td as sparse_td

if __name__ == '__main__':

    gamma = 0.9
    length = 20

    # Define environment and policy
    env = chain_walk.chain_walk(gamma, length)
    policy = chain_walk_policy.chain_walk_policy(length)

    # Set policy to optimal policy, i.e. move left if state < 10, move right if      state >= 10 (state index start with 0)
    p_mat = np.zeros([20, 2])# + 0.5
    p_mat[0:10, 0] = 1
    p_mat[10::, 1] = 1
    policy.set_policy(p_mat)

    # Get true value function for the policy
    vf = env.get_vf(policy)

    # Generate a sequence of 1000 noisy samples with 20 irrelavent features from     the environment
    n_noisy = 20
    n_samples = 1000
    n_iter = 500 #n_samples / length
    state_seq = []
    next_state_seq = []
    action_seq = []
    reward_seq = []
    for i in range(n_samples):
        # set to a new state every 50 iterations
        if i % n_iter == 0:
            env.set_cur_state(9 + i / n_iter)
            print(i / n_iter)
            state_seq.append(env.get_noisy_state(n_noisy))
        else:
            state_seq.append(sample[2])

       # Each sample is a tuple (action, reward, next state)
        sample = env.noisy_sample(policy, n_noisy)
        action_seq.append(sample[0])
        reward_seq.append(sample[1])
        next_state_seq.append(sample[2])

    # parameters for Elastic_TD
    # mu:       parameter for augmented Lagrangian
    # epsilon:  parameter for equility constraint
    # delta:    paramter for l1-norm and l2-norm
    # stop_ep:  parameter for stopping criteria (ADMM)
    mu = 1
    epsilon = 0.01
    delta = 0
    stop_ep = 0.01
    eta = 0.99

    # running Elastic_TD
    alg = elastic_td.Elastic_TD(n_samples, n_noisy + 3, gamma)
    beta = alg.run(mu, epsilon, delta, stop_ep, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
    #alg = sparse_td.Sparse_TD(n_samples - 1, n_noisy + 3, gamma)
    #beta = alg.run(mu, epsilon, np.array(state_seq), np.array(reward_seq))
    print(beta)

    # generate feature vectors for all states
    x = np.arange(length)
    phi_x = np.c_[np.ones(length), x, x ** 2]

    # calculate the aproximated value function
    beta_x = beta[0:3]
    V_x = np.dot(phi_x, beta_x)

    # generate the stationary distribution
    D = np.diag(env.get_stationary(policy))

    # calculate the MSE
    v = V_x - vf[:,0]
    loss1 = np.dot(np.dot(v.T, D), v)

    mu = 1
    epsilon = 0.01
    delta = 1
    stop_ep = 0.01
    eta = 0.99

    # running Elastic_TD
    alg = elastic_td.Elastic_TD(n_samples, n_noisy + 3, gamma)
    beta = alg.run(mu, epsilon, delta, stop_ep, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
    #alg = sparse_td.Sparse_TD(n_samples - 1, n_noisy + 3, gamma)
    #beta = alg.run(mu, epsilon, np.array(state_seq), np.array(reward_seq))
    print(beta)

    # generate feature vectors for all states
    x = np.arange(length)
    phi_x = np.c_[np.ones(length), x, x ** 2]

    # calculate the aproximated value function
    beta_x = beta[0:3]
    V_x = np.dot(phi_x, beta_x)

    # generate the stationary distribution
    D = np.diag(env.get_stationary(policy))

    # calculate the MSE
    v = V_x - vf[:,0]
    loss2 = np.dot(np.dot(v.T, D), v)

    print(loss1, loss2)
