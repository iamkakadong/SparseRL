import MDP.MDP
from MDP.chain_walk import *
import MDP.Policy
import numpy as np
import td.Elastic as Elastic
import td.elastic_td as elastic_td
import td.Elastic_fast as Elastic_fast

if __name__ == "__main__":
    gamma = 0.9
    length = 20

    #set parameters for solver
    epsilon = 0.01
    mu = 1
    alpha = 1
    eta = 0.9

    # Define environment and policy
    env = chain_walk(gamma, length)
    policy = chain_walk_policy(length)

    # Set policy to optimal policy, i.e. move left if state < 10, move right if state >= 10 (state index start with 0)
    p_mat = np.zeros([20, 2]) + 0.5
    #p_mat[0:10, 0] = 1
    #p_mat[10::, 1] = 1
    policy.set_policy(p_mat)

    # Get true value function for the policy
    vf = env.get_vf(policy)

    # Generate sequence of noisy samples for each state
    states = []
    next_states = []
    actions = []
    rewards = []

    n_noisy = 20
    n_samples = 1000
    for i in [9,10]:
        # set current state
        env.set_cur_state(i)
        state_seq = []
        action_seq = []
        reward_seq = []
        state_seq.append(env.get_noisy_state(n_noisy))

        #get trajectory samples
        for j in range(n_samples):
            sample = env.noisy_sample(policy, n_noisy)
            action_seq.append(sample[0])
            reward_seq.append(sample[1])
            state_seq.append(sample[2])

        # Add trajectory to the list
        states += state_seq[:-1]
        next_states += state_seq[1:]
        actions += action_seq
        rewards += reward_seq

    # solver = Elastic.Elastic_TD(gamma, mu, alpha, epsilon, states, next_states, rewards)
    # solver.ADMM()
    # print solver.theta
    # print solver.objs[-1]

    # # generate feature vectors for all states
    # x = np.arange(length)
    # phi_x = np.c_[np.ones(length), x, x ** 2]

    # # calculate the aproximated value function
    # beta_x = solver.theta[0:3]
    # V_x = np.dot(phi_x, beta_x)

    # # generate the stationary distribution
    # D = np.diag(env.get_stationary(policy))

    # # calculate the MSE
    # v = V_x - vf[:,0]
    # loss = np.dot(np.dot(v.T, D), v)

    # print(loss)

    solver = Elastic_fast.Elastic_TD(gamma, mu, alpha, eta, epsilon, states, next_states, rewards)
    solver.ADMM()
    print solver.theta
    print solver.objs[-1]

    # generate feature vectors for all states
    x = np.arange(length)
    phi_x = np.c_[np.ones(length), x, x ** 2]

    # calculate the aproximated value function
    beta_x = solver.theta[0:3]
    V_x = np.dot(phi_x, beta_x)

    # generate the stationary distribution
    D = np.diag(env.get_stationary(policy))

    # calculate the MSE
    v = V_x - vf[:,0]
    loss = np.dot(np.dot(v.T, D), v)

    print(loss)

    # alg = elastic_td.Elastic_TD(n_samples, n_noisy + 3, gamma)
    # beta = alg.run(mu, epsilon, alpha, 0.01, np.array(state_seq), np.array(reward_seq))
    # print(beta)

    # # calculate the aproximated value function
    # beta_x = beta[0:3]
    # V_x = np.dot(phi_x, beta_x)

    # # generate the stationary distribution
    # D = np.diag(env.get_stationary(policy))

    # # calculate the MSE
    # v = V_x - vf[:,0]
    # loss = np.dot(np.dot(v.T, D), v)

    # print(loss)
