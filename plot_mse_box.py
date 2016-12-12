import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    names = ['lstd', 'a0', 'a1e-1', 'a5e-1', 'a1']
    labels = ['lstd', 'L2', 'Elastic_0.1', 'Elastic_0.5', 'L1']
    noises = [20, 50, 100, 200, 500, 800]

    mse_all = {}
    for n_noise in noises:
        mse_all[n_noise] = []
    for name in names:
        with open('results/res_'+name+'.pickle', 'rb') as handle:
            res = pickle.load(handle)
            for n_noise in noises:
                mse_all[n_noise].append([res[(n_noise, i)][0] for i in range(10)])

    for n_noise in noises:
        mse_all[n_noise] = np.array(mse_all[n_noise]).T

    # plt.boxplot(mse_all[200], labels=labels)
    # plt.title('noise = 200', fontsize=15)
    # plt.tick_params(labelsize=15)

    # plt.boxplot(mse_all[500], labels=labels)
    # plt.title('noise = 500', fontsize=15)
    # plt.tick_params(labelsize=15)

    plt.boxplot(mse_all[800], labels=labels)
    plt.title('noise = 800', fontsize=15)
    plt.tick_params(labelsize=15)

    plt.show()