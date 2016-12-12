import MDP.chain_walk as chain_walk
import MDP.chain_walk_policy as chain_walk_policy
import numpy as np
import td.fast_elastic_td as elastic_td
import pickle

if __name__ == '__main__':

    gamma = 0.9
    length = 20
    n_samples = 1000

    # Define environment and policy
    env = chain_walk.chain_walk(gamma, length)
    policy = chain_walk_policy.chain_walk_policy(length)

    # Set policy to optimal policy, i.e. move left if state < 10, move right if state >= 10 (state index start with 0)
    p_mat = np.zeros([20, 2])
    p_mat[0:10, 0] = 1
    p_mat[10::, 1] = 1
    policy.set_policy(p_mat)

    # load samples
    with open('samples.pickle') as fin:
        data = pickle.load(fin)

    results = dict()
    NOISE = [20, 50, 100, 200, 500, 800]
    for n_noisy in NOISE:
        for i in range(10):
            print n_noisy, i
            state_seq = data[(n_noisy, i)][0]
            next_state_seq = data[(n_noisy, i)][1]
            reward_seq = data[(n_noisy, i)][2]

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

            # parameters for Elastic_TD
            # mu:       parameter for augmented Lagrangian
            # epsilon:  parameter for equility constraint
            # delta:    paramter for l1-norm and l2-norm
            # stop_ep:  parameter for stopping criteria (ADMM)
            mu = 10
            epsilon = 0.01
            stop_ep = 0.01
            eta = 0.5

            # running elastic
            delta = 0
            alg = elastic_td.Elastic_TD(n_samples, n_noisy + 3, gamma)
            beta = alg.run(mu, epsilon, delta, stop_ep, eta, np.array(state_seq), np.array(next_state_seq), np.array(reward_seq))
            print(beta)

            # mse, truth, pred = env.compute_mse(policy, theta, n_noisy, mc_iter=1000, restart=200)
            mse, truth, pred = env.compute_mse(policy, beta, n_noisy, mc_iter=1000, restart=200)
            print mse

            results[(n_noisy, i)] = [mse, beta]

    with open('results.pickle', 'wb') as fout:
        pickle.dump(results, fout)
