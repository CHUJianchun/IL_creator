import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def regress_2(t, p, solubility_list, title):
    feature_list = np.array([t, p, p ** 2]).T
    model = LinearRegression()
    model.fit(feature_list, solubility_list)
    solubility_333 = model.predict(np.array((333, 101, 101 ** 2)).reshape(1, -1))
    solubility_383 = model.predict(np.array((383, 101, 101 ** 2)).reshape(1, -1))
    solubility_diff = solubility_333 - solubility_383
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t, p, solubility_list, c='r', marker='o')
    ax.scatter(333, 101, solubility_333, c='#1f77b4', marker='s')
    ax.scatter(383, 101, solubility_383, c='#1f77b4', marker='s')
    plt.savefig('Graph_Analyze/property_figure/' + title + '.png')
    plt.close()
    return solubility_diff


def regress_1(t, p, solubility_list, title):
    feature_list = np.array([t, p]).T
    model = LinearRegression()
    model.fit(feature_list, solubility_list)
    solubility_333 = model.predict(np.array((333, 101)).reshape(1, -1))
    solubility_383 = model.predict(np.array((383, 101)).reshape(1, -1))
    solubility_diff = solubility_333 - solubility_383
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t, p, solubility_list, c='r', marker='o')
    ax.scatter(333, 101, solubility_333, c='#1f77b4', marker='s')
    ax.scatter(383, 101, solubility_383, c='#1f77b4', marker='s')
    plt.savefig('Graph_Analyze/property_figure/' + title + '.png')
    plt.close()
    return solubility_diff


if __name__ == '__main__':
    t = np.array([333.15,
                  343.15,
                  353.15,
                  363.15
                  ])

    p1 = np.array([97.5,
                   102.0,
                   109.5,
                   118.5
                   ])

    p2 = np.array([99.5,
                   100.8,
                   102,
                   103.5
                   ])

    s1 = np.array([0.014462605,
                   0.014343885,
                   0.013990653,
                   0.013664074
                   ])

    s2 = np.array([0.013267322,
                   0.013406816,
                   0.013545617,
                   0.013654696
                   ])

    sdiff1_2 = regress_2(t, p1, s1, '[EDiMIM][TOS]_2')
    sdiff2_2 = regress_2(t, p2, s2, '[EMIM][TOS]_2')
    sdiff1_1 = regress_1(t, p1, s1, '[EDiMIM][TOS]_1')
    sdiff2_1 = regress_1(t, p2, s2, '[EMIM][TOS]_1')
