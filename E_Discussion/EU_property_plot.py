import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
#  CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'


def draw_plots():
    with open('InputData/data_frame_list.data', 'rb') as f_:
        data = pickle.load(f_)
    with open('InputData/gp_dataset_list.data', 'rb') as f_:
        pickled = pickle.load(f_)
    dataframe = [np.array(d[6]) for d in data]
    idx = 0
    for i in range(len(dataframe)):
        item = dataframe[i]
        if len(item) > 5:
            if np.max(item[:, 0]) - np.min(item[:, 0]) >= 10:
                if np.min(item[:, 1] - 101.) <= (np.max(item[:, 1]) - np.min(item[:, 1])):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    p2 = item[:, 1] ** 2
                    model = LinearRegression()
                    model.fit(np.array([item[:, 0], item[:, 1], p2]).T, item[:, 2])
                    solubility_333 = model.predict(np.array((333, 101, 101 ** 2)).reshape(1, -1))
                    solubility_383 = model.predict(np.array((383, 101, 101 ** 2)).reshape(1, -1))

                    ax.scatter(item[:, 0], item[:, 1], item[:, 2], c='r', marker='o')
                    ax.scatter(333, 101, pickled[idx][6], c='b', marker='^')
                    ax.scatter(383, 101, pickled[idx][7], c='b', marker='^')
                    ax.scatter(333, 101, solubility_333, c='#1f77b4', marker='s')
                    ax.scatter(383, 101, solubility_383, c='#1f77b4', marker='s')
                    idx += 1
                    plt.savefig('Graph_Analyze/property_figure/' + str(i) + '.png')
                    plt.close()


def get_sdiff():
    with open('InputData/data_frame_list.data', 'rb') as f_:
        data = pickle.load(f_)
    with open('InputData/gp_dataset_list.data', 'rb') as f_:
        pickled = pickle.load(f_)
    dataframe = [np.array(d[6]) for d in data]
    idx = 0
    s_diff = []
    list_333 = []
    list_383 = []
    for i in range(len(dataframe)):
        item = dataframe[i]
        if len(item) > 5:
            if np.max(item[:, 0]) - np.min(item[:, 0]) >= 10:
                if np.min(item[:, 1] - 101.) <= (np.max(item[:, 1]) - np.min(item[:, 1])):
                    if 'NH' not in data[i][0] and 'NH' not in data[i][3]:
                        if min(item[:, 1]) < 130:
                            model = LinearRegression()
                            model.fit(
                                np.array([item[:, 0], item[:, 1], item[:, 1] ** 2]).T, item[:, 2])
                            solubility_333 = model.predict(np.array((333, 101, 101 ** 2)).reshape(1, -1))
                            solubility_383 = model.predict(np.array((383, 101, 101 ** 2)).reshape(1, -1))
                            list_333.append(solubility_333)
                            list_383.append(solubility_383)
                            s_diff.append([i, (solubility_333 - solubility_383)[0]])
    return np.array(list_333), np.array(list_383), np.array(s_diff)


if __name__ == '__main__':
    draw_plots()
    # a, b, c = get_sdiff()
