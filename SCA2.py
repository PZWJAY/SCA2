#   coding=utf-8
"""-----------------------------------------------------------------------------------------------------
1.Name      ：SCA2.py
2.Function  ：Apply SCA2 to cluster data
3.Author    ：by pzwjay, at 5.22/2019
4.Language  : Python<3.6.5>
5.Packages  ：numpy<1.14.3>, sklearn<0.20.2>, scipy<1.1.0>, networkx<2.1>,  matplotlib<1.5.0>
--------------------------------------------------------------------------------------------------------"""
import PublicFunctions
import Algorithms
import math
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn import preprocessing
import copy
from scipy import stats
import sys
import networkx as nx
import time
from collections import Counter
import newrb
sys.path.append(r'..\\')


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


def labeling(datapoints, bound, lbest_index, order):
    """---------------------------------------------------------------------------------------
    :function:  label the particle
    :parameter: datapoints  : original data points
                bound       : use to indicate whether the particle is merged with its leader, '1' means merged, '0' means no
                lbest_index : current leader in lbest
                order       : the order of particles based on original density
    :return:   point_labels : the label of each particle
               k            : number of clusters
    ------------------------------------------------------------------------------------------"""
    G = nx.Graph()
    n = len(datapoints)

    for i in range(n):
        if bound[i]:
            G.add_edge(i, lbest_index[i])
        else:
            G.add_node(i)
    # end-for
    clusters = nx.connected_components(G)
    labels = [-1 for i in range(n)]
    cnt = 0
    for cluster in clusters:
        if len(cluster) < 4:
            continue
        for ind in cluster:
            labels[ind] = cnt
        cnt += 1

    whole_neibrs = calculate_neighborhood(datapoints, n)

    for i in range(n):
        if labels[order[i][0]] == -1:
            j = 1
            while j < n and labels[whole_neibrs[order[i][0]][j]] == -1:
                j += 1
            labels[order[i][0]] = labels[whole_neibrs[order[i][0]][j]]
    # end-for
    return labels, cnt


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
                 "data/D31.txt": 0.5, "data/dim512.txt": 0.40, "data/iris.txt": 0.09, "data/wdbc.txt": 0.09,
                 "data/seeds.txt": 0.10, "data/segmentation_all.txt": 0.16, "data/ecoli.txt": 0.07,
                 "data/appendicitis.txt": 0.12}

    spread_dict = {"data/aggregation.txt": 1.8, "data/flame.txt": 2.2, "data/DS850.txt": 0.3, "data/R15.txt": 0.5,
                   "data/D31.txt": 1.0, "data/dim512.txt": 0.40, "data/iris.txt": 0.10, "data/wdbc.txt": 0.08,
                   "data/seeds.txt": 0.20, "data/segmentation_all.txt": 0.15, "data/ecoli.txt": 0.09,
                   "data/appendicitis.txt": 0.18}
    # use for drawing figures
    num_dict = {"data/aggregation.txt": 7, "data/flame.txt": 3, "data/DS850.txt": 5, "data/R15.txt": 15,
                "data/D31.txt": 31, "data/iris.txt": 0.10, "data/wdbc.txt": 0.08, "data/seeds.txt": 0.20,
                "data/segmentation_all.txt": 0.15, "data/ecoli.txt": 0.09}

    T = 100  # total time steps
    vmax = [0.0 for i in range(len(datapoints[0]))]
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
        axis_range[i][0] = min_val - 1
        axis_range[i][1] = max_val + 1

    np.random.seed(run)  # seed for random

    X = np.array(datapoints)  # record particles" position
    datapoints = X.tolist()
    n = len(X)                # data set scale
    d = len(X[0])             # dimension

    # calculate the density of particle using KDE
    bandwidth = band_dict[dataname]  # select bandwidth
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X.copy())
    f = [math.exp(a) for a in kde.score_samples(X)]  # densities of datapoints in KDE

    # train the RBF network
    t = np.array([[var] for var in f])
    net = newrb.designrb(X, t, 0, spread_dict[filename], round(0.1*n), n)
    print("Building RBF network is done!")

    for i in range(n):
        f[i] = net(X[i].reshape((1, d)))

    order = sorted(enumerate(f), key=lambda x: x[1], reverse=True)  # outliers' assignment order
    pbest = np.array(X)  # pbest
    pbest_density = copy.deepcopy(f)    # pbest's density
    # calculate leader for each particle
    neibrs_indices = calculate_neighborhood(pbest, n)  # neighbors for each data point
    lbest_index = [i for i in range(n)]  # the lbest's index
    lbest = np.array(X)  # lbest's current position
    k = 2  # number of leader that each particle almost can follow
    neibrs = []  # store the leader set for each particle
    for i in range(n):
        tmp = []
        cnt = 0
        for j in range(1, n):
            if f[i] < f[neibrs_indices[i][j]]:
                tmp.append(neibrs_indices[i][j])
                cnt += 1
            if cnt == k or j == n - 1:
                if not tmp:
                    tmp.append(neibrs_indices[i][0])
                neibrs.append(tmp)
                lbest_index[i] = tmp[0]  # select the first particle in its leader set as the initial leader
                lbest[i] = np.array(X[lbest_index[i]])  # obtain the position of lbest
                break
    # accquire the mean distance between particle and its leader, and the distance list of each pair
    avg_distance, distance_list = PublicFunctions.getAvgerageDistance(lbest_index, datapoints)
    bound = [0] * n     # use to indicate whether the particle is merged with its leader, '1' means merged, '0' means no

    velocity = np.array([np.array([0.0 for i in range(0, d)]) for j in range(0, n)])  # initialize the velocity to 0.0
    w = [1.0] * n   # initial inertial weight to 1.0

    for t in range(T):  # start iteration
        pbest_Mat = pbest - X
        lbest_Mat = lbest - X
        # update velocity and position
        for i in range(n):
            # if the particle is merged with its leader, it do not need to fly, just change its position to its leader
            if bound[i]:
                continue
            for j in range(d):
                r1 = np.random.random()
                r2 = np.random.random()
                # update velocity
                velocity[i][j] = w[i]*velocity[i][j] + c1 * r1 * (pbest_Mat[i][j]) + c2 * r2 * (lbest_Mat[i][j])
                # velocity clamped
                # v = vmax[j] - vmin[j]
                if velocity[i][j] > vmax[j]:
                    velocity[i][j] = vmax[j]
                elif velocity[i][j] < -vmax[j]:
                    velocity[i][j] = -vmax[j]
                X[i][j] = X[i][j] + velocity[i][j]  # calculate trial position
            # use RBFnn to estimate the density of the trail position
            new_density = net(X[i].reshape((1, d)))
            w[i] = new_density/pbest_density[i]
            # if the trial position is not bad than pbest, update the position to trail position
            f[i] = new_density
            if new_density >= pbest_density[i]:
                continue
            delta = ((pbest_density[i] - new_density)/pbest_density[i])*10   # calculate delta
            # according to the density difference between trail and pbest, adopt N~(0, 1/np.sqrt(2*pi)) to calculate
            # probability
            prob = stats.norm(0, 1/np.sqrt(2*np.pi)).pdf(delta)
            rand = np.random.random()   # generate a random number
            if rand > prob:     # if the random number is higher than the probability, the particle return to pbest
                velocity[i] = 0.0   # velocity reset to 0.0
                X[i] = pbest[i]     # position update to pbest
                f[i] = pbest_density[i]     # current density update to pbest's density
                ind = (neibrs[i].index(lbest_index[i]) + 1) % len(neibrs[i])    # select the next particle as leader
                lbest_index[i] = neibrs[i][ind]
        # end-for

        # update pbest in order
        for i in range(n):
            if bound[order[i][0]]:
                pbest[order[i][0]] = np.array(pbest[lbest_index[order[i][0]]])
            # update pbest
            elif f[order[i][0]] > pbest_density[order[i][0]]:
                pbest_density[order[i][0]] = f[order[i][0]]
                pbest[order[i][0]] = np.array(X[order[i][0]])

        # end-for
        # jude whether the particle should merged to its leader, if yes, update its pbest to its leader
        mark = [0] * n  # mark whether the particle should merge with its leader
        for i in range(n):
            if not bound[i] and PublicFunctions.getDistance(pbest[i], pbest[lbest_index[i]]) <= avg_distance:
                mark[i] = 1
        # end-for
        for i in range(n):
            if mark[order[i][0]] or bound[order[i][0]]:
                pbest[order[i][0]] = np.array(pbest[lbest_index[order[i][0]]])
                X[order[i][0]] = np.array(X[lbest_index[order[i][0]]])
                bound[order[i][0]] = 1
            # update the lbest's position
            lbest[order[i][0]] = np.array(X[lbest_index[order[i][0]]])
        # end-for

        if t == T - 1:
            labels, k = labeling(datapoints, bound, lbest_index, order)

            # Draw the final clustering result
            # PublicFunctions.drawClusteringResultGraph(pl, datapoints, labels, k, axis_range)
            # pl.show()
    # end-for
    return labels, k


if __name__ == "__main__":
    # parameters for compared algorithms
    dbscan_radius = {"data/aggregation.txt": 1.4, "data/flame.txt": 1.25, "data/DS850.txt": 0.4, "data/R15.txt": 0.7,
                     "data/D31.txt": 1.1, "data/dim512.txt": 0.36, "data/iris.txt": 0.12, "data/wdbc.txt": 0.46,
                     "data/seeds.txt": 0.24, "data/segmentation_all.txt": 0.15, "data/ecoli.txt": 0.2,
                     "data/appendicitis.txt": 0.3}
    dbscan_num = {"data/aggregation.txt": 7, "data/flame.txt": 8, "data/DS850.txt": 9, "data/R15.txt": 30,
                  "data/D31.txt": 48, "data/dim512.txt": 2, "data/iris.txt": 5, "data/wdbc.txt": 38,
                  "data/seeds.txt": 16, "data/segmentation_all.txt": 2, "data/ecoli.txt": 22,
                  "data/appendicitis.txt": 11}
    kmeans_num = {"data/aggregation.txt": 6, "data/flame.txt": 4, "data/DS850.txt": 5, "data/R15.txt": 10,
                  "data/D31.txt": 7, "data/dim512.txt": 16, "data/iris.txt": 3, "data/wdbc.txt": 2, "data/seeds.txt": 3,
                  "data/segmentation_all.txt": 4, "data/ecoli.txt": 5, "data/appendicitis.txt": 5}
    hac_num = {"data/aggregation.txt": 6, "data/flame.txt": 4, "data/DS850.txt": 5, "data/R15.txt": 15,
               "data/D31.txt": 31, "data/dim512.txt": 16, "data/iris.txt": 3, "data/wdbc.txt": 2, "data/seeds.txt": 3,
               "data/segmentation_all.txt": 4, "data/ecoli.txt": 8, "data/appendicitis.txt": 7}
    optics_radius = {"data/aggregation.txt": 1.40, "data/flame.txt": 1.3, "data/DS850.txt": 0.4, "data/R15.txt": 0.55,
                     "data/D31.txt": 0.95, "data/dim512.txt": 0.36, "data/iris.txt": 0.12, "data/wdbc.txt": 0.49,
                     "data/seeds.txt": 0.24, "data/segmentation_all.txt": 0.15, "data/ecoli.txt": 0.26,
                     "data/appendicitis.txt": 0.3}
    optics_num = {"data/aggregation.txt": 6, "data/flame.txt": 7, "data/DS850.txt": 8, "data/R15.txt": 11,
                  "data/D31.txt": 34, "data/dim512.txt": 22, "data/iris.txt": 4, "data/wdbc.txt": 50, "data/seeds.txt": 15,
                  "data/segmentation_all.txt": 1, "data/ecoli.txt": 40, "data/appendicitis.txt": 10}

    # 画图的图尺寸
    range_dict = {"data/aggregation.txt": [[2, 38], [0, 35]], "data/flame.txt": [[-2, 16], [12, 30]],
                  "data/R15.txt": [[2, 18], [2, 18]], "data/D31.txt": [[0, 35], [0, 35]],
                  "data/DS850.txt": [[-1, 5], [-0.5, 6.5]]}

    filename, data_set_type = PublicFunctions.select_file()
    datapoints, labels_true = PublicFunctions.readRawDataFromFile(filename)  # Read data from file
    if data_set_type == '2' or filename == "data/dim512.txt":
        min_max_scaler = preprocessing.MinMaxScaler()
        datapoints = min_max_scaler.fit_transform(datapoints)
    choice = input("""select algorithm:
        SCA2-------------1
        SCA--------------2
        K means----------3
        HAC--------------4
        DBSCAN-----------5
        OPTICS-----------6
        ....>>>""")
    times = 1 if choice not in ["1", "2", "3"] else 20
    result = []
    mean = []
    var = []
    whole_time = 0.0
    for i in range(times):
        if choice == "1":
            print("This is the %d run" % (i + 1))
            start = time.time()
            points_labels, k = PSOClusteringAlgorithm(datapoints, filename, i)
            end = time.time()
            whole_time += end-start
        elif choice == "2":
            print("This is the %d run" % (i + 1))
            start = time.time()
            points_labels, k = Algorithms.SCA_clustering(datapoints, filename, i)
            end = time.time()
            whole_time += end-start
        elif choice == "3":
            print("This is the %d run" % (i + 1))
            start = time.time()
            points_labels, k = Algorithms.kmeans(kmeans_num[filename], datapoints, i)  # k-means
            end = time.time()
            whole_time += end-start
        elif choice == "4":
            start = time.time()
            points_labels, k = Algorithms.agglomerativeClustering(hac_num[filename], datapoints)  # HAC
            end = time.time()
            whole_time += end-start
        elif choice == "5":
            start = time.time()
            points_labels, k = Algorithms.dbscan(dbscan_radius[filename], dbscan_num[filename], datapoints)
            end = time.time()
            whole_time += end-start
            sum = Counter(points_labels)
            print("Number of noises detected by DBSCAN:", sum[-1])
        elif choice == "6":
            start = time.time()
            points_labels, k = Algorithms.OPTICS(optics_radius[filename], optics_num[filename], datapoints)
            end = time.time()
            whole_time += end-start
            sum = Counter(points_labels)
            print("Number of noises detected by OPTICS:", sum[-1])
        else:
            print("No algorithm has been matched")
            exit(1)
        points_labels = np.array(points_labels)
        tmp = PublicFunctions.calValidator(datapoints, points_labels, labels_true, k)
        # PublicFunctions.drawClusteringResultGraph(pl, datapoints, points_labels, k, range_dict[filename])
        # pl.show()
        result.append(tmp)
    # end-for
    # calculate mean and variance
    result = np.array(result)
    for i in range(len(result[0])):
        mean.append(round(result[:, i:i + 1].mean(), 4))
        var.append(round(result[:, i:i + 1].var(), 4))
    # end-for
    print("Average time is: %f" % (whole_time/times))
    print("K mean : %f ; var : %f" % (mean[0], var[0]))
    print("F-measure mean : %f ; var : %f" % (mean[1], var[1]))
    print("NMI mean : %f ; var : %f" % (mean[2], var[2]))
    print("ARI mean : %f ; var : %f" % (mean[3], var[3]))
    print(' ')
