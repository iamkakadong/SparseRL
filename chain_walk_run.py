import MDP.chain_walk as chain_walk
import MDP.chain_walk_policy as chain_walk_policy
import numpy as np
import td.fast_elastic_td as elastic_td
import td.elastic_td as elastic
import td.lstd as lstd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    gamma = 0.9
    length = 20

    # Define environment and policy
    env = chain_walk.chain_walk(gamma, length)
    policy = chain_walk_policy.chain_walk_policy(length)

    # Set policy to optimal policy, i.e. move left if state < 10, move right if      state >= 10 (state index start with 0)
    p_mat = np.zeros([20, 2]) #+ 0.5
    p_mat[0:10, 0] = 1
    p_mat[10::, 1] = 1
    policy.set_policy(p_mat)

    # Get true value function for the policy
    vf = env.get_vf(policy)

    # Generate a sequence of 1000 noisy samples with 20 irrelavent features from     the environment
    n_noisy = 500
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
            state_seq.append(env.get_noisy_state(n_noisy))
        else:
            state_seq.append(sample[2])

       # Each sample is a tuple (action, reward, next state)
        sample = env.noisy_sample_corr(policy, n_noisy)
        action_seq.append(sample[0])
        reward_seq.append(sample[1])
        next_state_seq.append(sample[2])

    # running lstd
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
    print theta

    # generate feature vectors for all states
    x = np.arange(length)
    phi_x = np.c_[np.ones(length), x, x ** 2]

    # calculate the aproximated value function
    beta_x = theta[0:3]
    V_y = np.dot(phi_x, beta_x)

    # parameters for Elastic_TD
    # mu:       parameter for augmented Lagrangian
    # epsilon:  parameter for equility constraint
    # delta:    paramter for l1-norm and l2-norm
    # stop_ep:  parameter for stopping criteria (ADMM)
    mu = 10
    epsilon = 0.01
    stop_ep = 0.01
    eta = 0.5

   # # running putong Elastic_TD
   # alg = elastic.Elastic_TD(n_samples, n_noisy + 3, gamma)
   # beta_putong = alg.run(mu, epsilon, delta, stop_ep, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
   # print(beta_putong)

   # # generate feature vectors for all states
   # x = np.arange(length)
   # phi_x = np.c_[np.ones(length), x, x ** 2]

   # # calculate the aproximated value function
   # beta_x = beta_putong[0:3]
   # V_x = np.dot(phi_x, beta_x)

   # # generate the stationary distribution
   # D = np.diag(env.get_stationary(policy))

   # # calculate the MSE
   # v = V_x - vf[:,0]
   # loss2 = np.dot(np.dot(v.T, D), v)

   # print loss2

   # # running l2
   # delta = 0.0
   # alg = elastic_td.Elastic_TD(n_samples, n_noisy + 3, gamma)
   # beta_l2 = alg.run(mu, epsilon, delta, stop_ep, eta, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
   # print(beta_l2)

   # # running Elastic_TD
   # delta = 0.5
   # alg = elastic_td.Elastic_TD(n_samples, n_noisy + 3, gamma)
   # beta_elas = alg.run(mu, epsilon, delta, stop_ep, eta, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
   # print(beta_elas)

    # running l1
    delta = 1
    alg = elastic_td.Elastic_TD(n_samples, n_noisy + 3, gamma)
    beta_l1 = alg.run(mu, epsilon, delta, stop_ep, eta, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
    print(beta_l1)

    mse, truth, pred = env.compute_mse(policy, theta, n_noisy, mc_iter=1000, restart=200)
    mse_l1, truth, pred = env.compute_mse(policy, beta_l1, n_noisy, mc_iter=1000, restart=200)

    print mse, mse_l1

    V_x = np.dot(phi_x, beta_l1[0:3])
    plt.plot(V_x)
    plt.plot(V_y)
    plt.plot(vf)
    plt.show()
