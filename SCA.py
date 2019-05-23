#   coding=utf-8
"""-----------------------------------------------------------------------------------------------------
1.Name：SCA.py
2.Function：Apply SCA to cluster data points
3.Author  ：by pzwjay, at 9.13/2018
5.第三方库：numpy<1.14.3>, sklearn<0.20.2>, matplotlib<1.5.0>
--------------------------------------------------------------------------------------------------------"""
import math
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sys import path
path.append(r"..\\")


def calculate_neighborhood(X, size, mode="kneighbors"):
    """---------------------------------------------------------------------------------------
    :function:  acquire the neighborhoods of particles
    :parameter: X               : data points
                size            : number of particles in neighborhoods
                mode            : "kneighbors" and "epsilon neighbors"
    :return:    neibrs_indices  : index of particles in neighbors
    ------------------------------------------------------------------------------------------"""
    if mode == "kneighbors":
        neibrs = NearestNeighbors(n_neighbors=size, algorithm="ball_tree").fit(X)
        neibrs_distances, neibrs_indices = neibrs.kneighbors(X)
    else:   # "epsilon neighbors"
        neibrs = radius_neighbors_graph(X, radius=size, mode="distance")
        neibrs_indices = [[i] for i in range(0, len(X))]
        nonzero_indices = neibrs.nonzero()
        for x, y in zip(nonzero_indices[0], nonzero_indices[1]):
            neibrs_indices[x].append(y)
    return neibrs_indices

def labeling(datapoints, neibrs):
    """---------------------------------------------------------------------------------------
    :function:  label the particle
    :parameter: datapoints  : original data points
                neibrs      : neighbors of particles
    :return:   point_labels : the label of each particle
               k            : number of clusters
    ------------------------------------------------------------------------------------------"""
    k = 0
    point_labels = [-1 for i in range(0, len(neibrs))]
    label_dict = {}

    def _recursion(target, lab):
        for val in neibrs[target]:
            if point_labels[val] == -1:
                point_labels[val] = lab
                _recursion(val, lab)

    for ls in neibrs:
        if point_labels[ls[0]] == -1:
            k += 1
            point_labels[ls[0]] = k
            for i in ls:
                if point_labels[i] == -1:
                    point_labels[i] = k
                    _recursion(i, k)

    for i in range(len(point_labels)):
        if point_labels[i] not in label_dict:
            label_dict[point_labels[i]] = []
        label_dict[point_labels[i]].append(i)
    # end-for
    # dissolve cluster with particles less than 10 and reassign them to nearest cluster
    outliers = []
    for label in label_dict:
        if len(label_dict[label]) < 10:
            for j in label_dict[label]:
                outliers.append(j)
    # end-for
    whole_neibrs = calculate_neighborhood(datapoints, len(datapoints))
    for point in outliers:
        its_neibrs = whole_neibrs[point]
        for ner in its_neibrs:
            if ner not in outliers:
                point_labels[point] = point_labels[ner]
                break
        # end-for
    # end-for
    copy_label = point_labels.copy()
    copy_label.sort()
    copy_label = list(set(copy_label))
    for i in range(len(point_labels)):
        point_labels[i] = copy_label.index(point_labels[i])
    return point_labels, len(set(point_labels))

def PSOClusteringAlgorithm(datapoints, dataname, run):
    """---------------------------------------------------------------------------------------
    :function:  PSO clustering algorithm
    :parameter: datapoints: data points
                dataname:   data set's name
                run:  indicates which time is it and use to set the seed for random
    :return:    lables: the labels of data points assigned by the algorithm
                k:      number of cluster detected
    ------------------------------------------------------------------------------------------"""
    band_dict = {"data/aggregation.txt": 1.5, "data/flame.txt": 1.3, "data/DS850.txt": 0.3, "data/R15.txt": 0.3,
                 "data/D31.txt": 0.5, "data/iris.txt": 0.06, "data/wdbc.txt": 0.11, "data/seeds.txt": 0.13,
                 "data/segmentation_all.txt": 0.17, "data/ecoli.txt": 0.07, "data/dim512.txt": 0.4,
                 "data/appendicitis.txt": 0.16}

    T = 100  # total time steps
    vmax = [0.0 for i in range(len(datapoints[0]))]
    vmin = [0.0 for i in range(len(datapoints[0]))]
    axis_range = [[0.0, 0.0] for i in range(len(datapoints[0]))]

    c1 = 1.0  # acceleration constant for cognitive learning
    c2 = 0.5  # acceleration constant for social learning

    # obtain the range on each dimension and the maximum velocity on each dimension
    dp = np.array(datapoints)
    for i in range(len(dp[0])):
        tmp = dp[0:len(dp), i:i+1]
        max_val = tmp.max()
        min_val = tmp.min()
        if vmax[i] < (max_val - min_val) * 0.10:
            vmax[i] = (max_val - min_val) * 0.10
        axis_range[i][0] = min_val - 0.5
        axis_range[i][1] = max_val + 0.5

    np.random.seed(run)  # seed for random

    X = np.array(datapoints)  # record particles" position
    n = len(X)                # data set scale
    d = len(X[0])             # dimension

    X_velocity = np.array([np.array([0.0 for i in range(0, d)]) for j in range(0, n)])   # initial velocity to 0.0
    pbest = np.array(X)       # initial pbest
    pbest_density = np.array([0.0 for i in range(0, n)])
    lbest = np.array(X)
    lbest_index = [i for i in range(n)]

    # calculate the density of particle using KDE
    bandwidth = band_dict[dataname]    # select bandwidth of KDE
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X.copy())
    X_t_1 = X.copy()    # record the position of particles at last time step
    for t in range(T):
        X_old = X.copy()
        f = [math.exp(a) for a in kde.score_samples(X)]
        if t == 0:
            pbest_density = np.array(f)
            # calculate leader
            neibrs_indices = calculate_neighborhood(pbest, n)
            for i in range(n):
                for j in range(1, len(neibrs_indices[i])):
                    k = neibrs_indices[i][j]
                    if pbest_density[k] > f[i]:
                        lbest_index[i] = k
                        break
                # end-for
            # end-for
        pbest_old = pbest.copy()
        for i in range(n):
            # update pbest
            if f[i] > pbest_density[i]:
                pbest_density[i] = f[i]
                pbest[i] = np.array(X[i])
            # update lbest
            lbest[i] = np.array(X[lbest_index[i]])
        pbest_Mat = pbest - X
        lbest_Mat = lbest - X
        # update velocity and position
        for i in range(n):
            if (X[i] == X_t_1[i]).all() or (X[i] == pbest_old[i]).all():
                w = 0
            else:
                w = f[i] / pbest_density[i]
            for j in range(d):
                r1 = np.random.random()
                r2 = np.random.random()
                X_velocity[i][j] = w*X_velocity[i][j] + c1 * r1 * (pbest_Mat[i][j]) + c2 * r2 * (lbest_Mat[i][j])  # 更新速度
                v = vmax[j] - vmin[j]
                if X_velocity[i][j] > v:
                    X_velocity[i][j] = v
                elif X_velocity[i][j] < -v:
                    X_velocity[i][j] = -v
                X[i][j] = X[i][j] + X_velocity[i][j]
            new_density = math.exp(kde.score_samples([X[i]]))
            tmp_sum = new_density + pbest_density[i] + f[i]
            prob1 = pbest_density[i] / tmp_sum
            prob2 = f[i] / tmp_sum
            rand = np.random.random()
            if rand <= prob1:
                X[i] = pbest[i]
            elif rand <= (prob1+prob2):
                X[i] = X_old[i]
        # end-for
        X_t_1 = X_old.copy()
        if t == T - 1:
            neibrs_indices = calculate_neighborhood(pbest, bandwidth, "neibr")  # calculate neighbors
            labels, k = labeling(pbest, neibrs_indices)
    # end-for
    return labels, k
