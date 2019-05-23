#   coding=utf-8
"""-----------------------------------------------------------------------------------------------------
1.Name      ：Algorithms.py
2.Function  ：All compared algorithms are called from here
3.Author    ：by pzwjay, at 9.13/2018
4.Language  : Python<3.6.5>
5.Packages  ：numpy<1.14.3>, sklearn<0.20.2>, scipy<1.1.0>, networkx<2.1>,  matplotlib<1.5.0>
--------------------------------------------------------------------------------------------------------"""

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from pyclustering.cluster.optics import optics
import numpy as np
from SCA import PSOClusteringAlgorithm


def dbscan(eps, min_pts, datapoints):
    """--------------------------------------------------------------------------------
    :function: Apple DBSCAN to cluster data
    :parameter: eps       : the radius of a neighbourhood for each
                min_pts   : the at least min_pts within the eps neighbourhood of the points
                datapoints: datapoint to cluster
    :return: labels: labels for date points
    --------------------------------------------------------------------------------"""
    model = DBSCAN(eps=eps, min_samples=min_pts)
    res = model.fit_predict(datapoints)
    k = len(set(res)) - (1 if -1 in res else 0)
    return res, k


def kmeans(k, datapoints, run):
    """--------------------------------------------------------------------------------
    :function: Apply k-means to cluster data
    :parameter: k:  Number of cluster
                datapoints: datapoint to cluster
    :return: labels: labels for date points
    --------------------------------------------------------------------------------"""
    model = KMeans(n_clusters=k, random_state=run)
    res = model.fit_predict(datapoints)
    return res, k


def agglomerativeClustering(k, datapoints):
    """--------------------------------------------------------------------------------
    :function: Apply AgglomerativeClustering to cluster data
    :parameter: k:  Number of clusters
                datapoints: datapoint to cluster
    :return: labels: labels for date points
    --------------------------------------------------------------------------------"""
    model = AgglomerativeClustering(n_clusters=k)
    res = model.fit_predict(datapoints)
    return res, k


def OPTICS(radius, num, datapoints):
    """--------------------------------------------------------------------------------
    :function: Apple OPTICS to cluster data
    :parameter: eps       : the radius of a neighbourhood for each
                min_pts   : the at least min_pts within the eps neighbourhood of the points
                datapoints: datapoint to cluster
    :return: labels: labels for date points
    --------------------------------------------------------------------------------"""
    model = optics(datapoints, radius, num)
    model.process()
    clusters = model.get_clusters()
    labels = np.array([-1] * len(datapoints))
    k = 0
    for cluster in clusters:
        k += 1
        labels[np.array(cluster)] = k
    return labels, k


def SCA_clustering(datapoints, dataname, run):
    """--------------------------------------------------------------------------------
    :function: Apple SCA to cluster data
    :parameter: datapoints: datapoint to cluster
                dataname:   name of the file
                run:        indicates which time is it and use to set the seed for random
    :return: labels: labels for date points
    --------------------------------------------------------------------------------"""
    lables, k = PSOClusteringAlgorithm(datapoints, dataname, run)
    return lables, k