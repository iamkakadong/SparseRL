import numpy as np
from MDP.pendulum import pendulum
from MDP.pendulum_policy import pendulum_policy
import td.fast_elastic_td as elastic_td
import td.Elastic_fast as Elastic_fast

if __name__ == '__main__':

    gamma = 0.95
    dim = 20
    length = 1
    mass = 1
    sigma = 1
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

    # Generate a sequence of 1000 noisy samples with 20 irrelavent features from the environment
    n_noisy = 0
    n_samples = 1000
    state_seq = list()
    next_state_seq = list()
    action_seq = list()
    reward_seq = list()
    state = env.get_noisy_state(n_noisy)
    for i in range(n_samples):
        # Each sample is a tuple (action, reward, next state)
        state_seq.append(state)
        sample = env.noisy_sample(policy, n_noisy)
        action_seq.append(sample[0])
        reward_seq.append(sample[1])
        next_state_seq.append(sample[2])
        state = sample[2]

    # set parameters
    mu = 1
    epsilon = 0.01
    stop_ep = 0.01
    eta = 0.9

    # run l1
    delta = 1
    alg = elastic_td.Elastic_TD(n_samples, n_noisy + 2 * dim + 1, gamma)
    beta_l1 = alg.run(mu, epsilon, delta, stop_ep, eta, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
    print beta_l1

    # compute MSE
    loss_l1, truth_l1, pred_l1  = env.compute_mse(policy, beta_l1)
    print loss_l1
    print truth_l1[0:10]
    print pred_l1[0:10]

    # run elastic net
    delta = 0.5
    alg = elastic_td.Elastic_TD(n_samples, n_noisy + 2 * dim + 1, gamma)
    beta_elas = alg.run(mu, epsilon, delta, stop_ep, eta, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
    print beta_elas

    # compute MSE
    loss_elas, truth_elas, pred_elas  = env.compute_mse(policy, beta_elas)
    print loss_elas
    print truth_elas[0:10]
    print pred_elas[0:10]

    # run elastic net
    delta = 0
    alg = elastic_td.Elastic_TD(n_samples, n_noisy + 2 * dim + 1, gamma)
    beta_l2 = alg.run(mu, epsilon, delta, stop_ep, eta, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
    print beta_l2

    # compute MSE
    loss_l2, truth_l2, pred_l2  = env.compute_mse(policy, beta_l2)
    print loss_l2
    print truth_l2[0:10]
    print pred_l2[0:10]
