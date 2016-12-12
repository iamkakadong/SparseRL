import MDP.MDP
import MDP.Policy
import numpy as np
from MDP.pendulum import pendulum
from MDP.pendulum_policy import pendulum_policy
import td.Elastic_fast as Elastic_fast

if __name__ == '__main__':

    gamma = 0.95
    dim = 20
    length = 1
    mass = 1
    sigma = 0.1
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
    n_noisy = 20
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

    #set parameters for solver
    epsilon = 0.01
    mu = 1
    alpha = 1
    eta = 0.9

    solver = Elastic_fast.Elastic_TD(gamma, mu, alpha, eta, epsilon, state_seq, next_state_seq, reward_seq)
    solver.ADMM()
    print solver.theta
    print solver.objs[-1]
    loss_1, truth_1, pred_1 = env.compute_mse(policy, solver.theta, 100, 1000)

    alpha = 0.5
    solver = Elastic_fast.Elastic_TD(gamma, mu, alpha, eta, epsilon, state_seq, next_state_seq, reward_seq)
    solver.ADMM()
    print solver.theta
    print solver.objs[-1]
    loss_2, truth_2, pred_2 = env.compute_mse(policy, solver.theta, 100, 1000)

    alpha = 0
    solver = Elastic_fast.Elastic_TD(gamma, mu, alpha, eta, epsilon, state_seq, next_state_seq, reward_seq)
    solver.ADMM()
    print solver.theta
    print solver.objs[-1]
    loss_3, truth_3, pred_3 = env.compute_mse(policy, solver.theta, 100, 1000)

    print loss_1, loss_2, loss_3
    print truth_1[:10], pred_1[:10], truth_1[-10:], pred_1[-10:]
    print truth_2[:10], pred_2[:10], truth_2[-10:], pred_2[-10:]
    print truth_3[:10], pred_3[:10], truth_3[-10:], pred_3[-10:]