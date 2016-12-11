import numpy as np
from pendulum import pendulum
from pendulum_policy import pendulum_policy

if __name__ == '__main__':

    gamma = 0.95
    dim = 20
    length = 1
    mass = 1
    sigma = 0.01
    dt = 0.01
    penalty=0.01
    action_penalty=0.0

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
    action_seq = list()
    reward_seq = list()
    state_seq.append(env.get_noisy_state(n_noisy))
    for i in range(n_samples):
        # Each sample is a tuple (action, reward, next state)
        sample = env.noisy_sample(policy, n_noisy)
        action_seq.append(sample[0])
        reward_seq.append(sample[1])
        state_seq.append(sample[2])

