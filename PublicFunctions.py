#   coding=utf-8
'''-----------------------------------------------------------------------------------------------------
1.Name：PublicFunctions.py
2.Function：public modules
3.Authors：by pzwjay, at 5.22/2019
5.Packages：numpy<1.14.3>, sklearn<0.20.2>
--------------------------------------------------------------------------------------------------------'''
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict


def select_real_world_datasets():
    """--------------------------------------------------------------------------
    :function:   Select real-world datasets
    :returns:    filename:file name, type:string
    --------------------------------------------------------------------------"""
    filename = ""
    choice = input("""select a real world data set:
        iris-------------1
        WDBC-------------2
        Seeds------------3
        Segmentation-----4
        Ecoli------------5
        Appendicitis-----6
        ....>>>""")
    if choice == '1':
        filename = "data/iris.txt"
    elif choice == '2':
        filename = "data/wdbc.txt"
    elif choice == '3':
        filename = "data/seeds.txt"
    elif choice == '4':
        filename = "data/segmentation_all.txt"
    elif choice == '5':
        filename = 'data/ecoli.txt'
    elif choice == '6':
        filename = 'data/appendicitis.txt'
    else:
        print("input error!!")
        exit(1)
    return filename


def select_synthetic_datasets():
    """--------------------------------------------------------------------------
    :function:   select synthetic datasets
    :returns:    filename:file name, type:string
    --------------------------------------------------------------------------"""
    filename = ""
    choice = input("""select a synthetic data set:
        aggregation------1
        flame------------2
        DS850------------3
        R15--------------4
        D31--------------5
        DIM512-----------6
        ....>>>""")
    if choice == '1':
        filename = "data/aggregation.txt"
    elif choice == '2':
        filename = "data/flame.txt"
    elif choice == '3':
        filename = "data/DS850.txt"
    elif choice == '4':
        filename = "data/R15.txt"
    elif choice == '5':
        filename = "data/D31.txt"
    elif choice == '6':
        filename = "data/dim512.txt"
    else:
        print("input error!!")
        exit(1)
    return filename


def select_file():
    """--------------------------------------------------------------------------
    :function:   select which type of data set
    :returns:    filename:file name, type:string
    --------------------------------------------------------------------------"""
    filename = ""
    choice = input("""select a dataset:
        Synthetic data sets--1
        Real World data sets-2
        ....>>>""")
    if choice == '1':
        filename = select_synthetic_datasets()
    elif choice == '2':
        filename = select_real_world_datasets()
    else:
        print("input error!!")
        exit(1)
    return filename, choice


def readRawDataFromFile(filename):
    """--------------------------------------------------------------------------
    :function:   read data from file
    :parameter:  filename:file name, type:string
    :returns:    points: a list, each element is presented as tuple
                 labels: a list, labels of all points
    --------------------------------------------------------------------------"""
    readFile = open(filename, 'r')
    points = []
    labels = []
    for line in readFile.readlines():
        if ',' in line:
            data = line.split(',')
        else:
            data = line.split()
        ls = []
        for index, dim in enumerate(data):
            if index != len(data) - 1:
                ls.append(float(dim))
            else:
                labels.append(int(dim))
        # end-for
        points.append(tuple(ls))
    readFile.close()
    return points, labels


def getDistance(pt1, pt2):
    """-----------------------------------------------------------------------
    :function:   calculate the distance between two points
    :parameter:  pt1: point 1
                 pt2: point 2
    :returns:    distance:float，present the distance between two points
    ---------------------------------------------------------------------------"""
    res = 0
    for i in range(len(pt1)):
        res += pow(pt1[i] - pt2[i], 2)
    return pow(res, 0.5)


def getAvgerageDistance(leader, datapoints):
    """--------------------------------------------------------------------------
    :function:   calculate the mean distance between all pair of particle and their leaders
    :parameter:  leader：the index of leader
                 datapoints: data points
    :returns:    average_distance: the mean distance
                 distance_list: a list, each element is the distance between particle and its leader
    --------------------------------------------------------------------------"""
    distance = 0
    n = len(datapoints)
    distance_list = []
    for i in range(n):
        tmp = getDistance(datapoints[i], datapoints[leader[i]])
        distance_list.append(tmp)
        distance += tmp
    average_distance = distance / n
    return average_distance, distance_list


def calcluateFMeasure(labels_detect, labels_true):
    """----------------------------------------------------------------
    function: calculate metric F-measure
    parameters:  filename:   data file name
                 labels_detect:     all label of points
                 labels_true:   true labels of all points
    returns:     F_measure:  the value of F-measure
    -------------------------------------------------------------------"""
    true_partition = defaultdict(lambda: [])
    my_partition = defaultdict(lambda: [])
    for i in range(0, len(labels_detect)):
        my_partition[labels_detect[i]].append(i)
        true_partition[labels_true[i]].append(i)

    # calculate P matrix
    F_matrix = np.zeros((len(true_partition), len(my_partition)))
    size_true_Ci = []
    i = 0
    for key1 in true_partition.keys():
        j = 0
        for key2 in my_partition.keys():
            temp = set(my_partition[key2]).intersection(set(true_partition[key1]))
            P = len(temp)/len(my_partition[key2])
            R = len(temp)/len(true_partition[key1])
            if P == 0.0 and R == 0.0:
                F_matrix[i][j] = 0.0
            else:
                F_matrix[i][j] = 2*P*R/(P+R)
            j += 1
        size_true_Ci.append(len(true_partition[key1]))  # size of Ci. Ci is the true cluster
        i += 1

    n = len(labels_detect)
    F_measure = 0
    for i in range(0, len(true_partition)):
        max_FPi = max(F_matrix[i])
        F_measure += max_FPi*size_true_Ci[i]/n
    return F_measure


def calcNMI(labels_detect, labels_true):
    """----------------------------------------------------------------
    function: calculate the metric NMI
    parameters:  filename:   data file name
                 labels:     all label of points
    returns:     NMI:  the value of NMI
    -------------------------------------------------------------------"""
    NMI = normalized_mutual_info_score(labels_true, labels_detect)
    return NMI


def calcARI(labels_detect, labels_true):
    """----------------------------------------------------------------
    function: calculate the metric ARI
    parameters:  filename:   data file name
                 labels:     all label of points
    returns:     ARI:  the value of ARI
    -------------------------------------------------------------------"""
    ARI = adjusted_rand_score(labels_true, labels_detect)
    return ARI


def drawClusteringResultGraph(pl, points, label, labelnum, range0):
    """-------------------------------------------------------------------------------------
    :function:   draw figure
    :parameter:  pl      ：pylab
                 points  ：data points
                 label   : lable of each point
                 labelnum: number of clusters
                 range0  : range of the figure
    :return:     None
    ---------------------------------------------------------------------------------------"""
    x = [xx for (xx, yy) in points]
    y = [yy for (xx, yy) in points]
    cm = pl.get_cmap("RdYlGn")
    for i in range(0, len(points)):
        if label[i] >= 0:
            pl.plot(x[i], y[i], 'o', color=cm(label[i]*1.0/labelnum))   # color_map[label[i]]) cm(label[i]*1.0/labelnum))
        if label[i] < 0:
            pl.plot(x[i], y[i], 'x', color='k')

    # set range
    pl.xlim(range0[0][0], range0[0][1])
    pl.ylim(range0[1][0], range0[1][1])


def calValidator(datapoints, points_labels, labels_true, k):
    """----------------------------------------------------------------
    function: calculate the F-measure, NMI and ARI metrics
    parameters:  datapoints   : data points
                 points_labels: detected labels
                 labels_true  : true labels
                 k            : number of clusters
    returns:     ARI:  the value of ARI
    -------------------------------------------------------------------"""

    print("The clusters num is:%f" % k)
    f_measure = calcluateFMeasure(points_labels, labels_true)
    print("The F-Measure is:%f" % f_measure)
    '''
    AMI = calcAMI(filename, points_labels)
    print("The AMI is:%f" % AMI)
    '''
    NMI = calcNMI(points_labels, labels_true)
    print("The NMI is:%f" % NMI)
    ARI = calcARI(points_labels, labels_true)
    print("The ARI is:%f" % ARI)
    return [k, f_measure, NMI, ARI]


if __name__ == "__main__":
    print("Hello world")
