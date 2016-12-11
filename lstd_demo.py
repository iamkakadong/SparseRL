import MDP.MDP
from MDP.chain_walk import *
import MDP.Policy
import numpy as np
import td.Elastic as Elastic
import td.elastic_td as elastic_td
from td.lstd import lstd

if __name__ == "__main__":
    gamma = 0.9
    length = 20

    n_noisy = 500
    n_samples = 1000

    #set parameters for solver
    lam = 0.0
    dim = 3 + n_noisy
    epsilon = 0.01
    mu = 1
    alpha = 1

    # Define environment, policy, and agent
    env = chain_walk(gamma, length)
    policy = chain_walk_policy(length)
    agent = lstd(lam, dim, gamma)

    # Set policy to optimal policy, i.e. move left if state < 10, move right if state >= 10 (state index start with 0)
    # p_mat = np.zeros([20, 2]) + 0.5
    p_mat = np.zeros([20, 2])
    p_mat[0:10, 0] = 1
    p_mat[10::, 1] = 1
    policy.set_policy(p_mat)

    # Get true value function for the policy
    vf = env.get_vf(policy)

    # Set current state of environment to 0
    init_state = 9
    env.set_cur_state(init_state)

    # Generate a sequence of 1000 noisy samples with 20 irrelavent features from the environment
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

    # Learning
    agent.set_start(state_seq[0])
    prev_state = state_seq[0]
    for i in range(len(reward_seq)):
        agent.update_V(prev_state, state_seq[i + 1], reward_seq[i])
        prev_state = state_seq[i + 1]

    # Examine result
    theta = agent.get_theta()
    print theta

    '''
    solver = Elastic.Elastic_TD(gamma, mu, alpha, epsilon, state_seq, reward_seq)
    solver.ADMM()
    print solver.theta
    print solver.objs[-1]
    '''

    mse, truth, pred = env.compute_mse(policy, theta, n_noisy)

    '''
    alg = elastic_td.Elastic_TD(n_samples-1, n_noisy + 3, gamma)
    beta = alg.run(mu, epsilon, alpha, 0.01, np.array(state_seq), np.array(reward_seq))
    print(beta)

    # calculate the aproximated value function
    beta_x = beta[0:3]
    V_x = np.dot(phi_x, beta_x)

    # generate the stationary distribution
    D = np.diag(env.get_stationary(policy))

    # calculate the MSE
    v = V_x - vf[:,0]
    loss = np.dot(np.dot(v.T, D), v)

    print(loss)
    '''
