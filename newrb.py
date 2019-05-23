#   coding=utf-8
"""-----------------------------------------------------------------------------------------------------
1.Name      ：newrb.py
2.Function  ：re-implement the newrb() function in Matlab using Python. We only implement the designrb()
              function in the newrb.m.
3.Author    ：by Yigui Yuan and pzwjay, at 5.22/2019
4.Language  : Python<3.6.5>
5.Packages  ：numpy<1.14.3>, sklearn<0.20.2>, scipy<1.1.0>
--------------------------------------------------------------------------------------------------------"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')


def solvelin2(p, t):
    p = np.transpose(p)
    pr, pc = p.shape
    temp = np.row_stack((p, np.ones([1, pc])))
    x = np.dot(np.transpose(t), np.linalg.pinv(temp))
    w = x[:, range(pr)]
    b = x[:, pr][:, np.newaxis]
    return w, b


def designrb(p, t, goal=0.0, spread=1.0, MN=-1, df=25):
    p = np.array(p)
    t = np.array(t)
    q, r = p.shape
    s2 = t.shape[1]

    # RADIAL BASIS LAYER OUTPUTS
    squre_tmp = pdist(p)
    P = radbas(squareform(squre_tmp), spread)
    PP = np.sum(P ** 2, axis=1, keepdims=True)
    dd = np.sum(t ** 2, axis=0, keepdims=True)

    # CALCULATE "ERRORS" ASSOCIATED WITH VECTORS
    e = np.matmul(P, t) ** 2 / np.matmul(PP, dd)

    # PICK VECTOR WITH MOST "ERROR"
    pick = find_large_row(e)
    used = []
    left = list(range(q))
    used.append(left[pick])
    wj = P[pick][np.newaxis, :]
    P = np.delete(P, pick, 0)
    del left[pick]

    if MN == -1 or MN > q:
        MN = q

    for k in range(MN - 1):

        # CALCULATE "ERRORS" ASSOCIATED WITH VECTORS
        a = np.matmul(wj, np.transpose(P)) / np.matmul(wj, np.transpose(wj))
        P = P - np.transpose(a) * wj
        PP = np.sum(P ** 2, axis=1, keepdims=True)
        e = np.matmul(P, t) ** 2/np.matmul(PP, dd)

        # PICK VECTOR WITH MOST "ERROR"
        pick = find_large_row(e)
        used.append(left[pick])
        wj = P[pick][np.newaxis, :]
        P = np.delete(P, pick, 0)
        del left[pick]

        # CALCULATE ACTUAL ERROR
        W1 = p[used]
        a1 = radbas(cdist(p, W1), spread)
        w2, b2 = solvelin2(a1, t)
        a2 = np.dot(w2, np.transpose(a1)) + np.dot(b2, np.ones((1, q)))
        MSE = mean_squared_error(t, np.transpose(a2))

        if (k + 2) % df == 0:
            print("Epoch:", k + 2, "MSE:%.10f" % MSE)

        # CHECK ERROR
        if MSE < goal:
            break

    def func(X):
        a1 = radbas(cdist(X, W1), spread)
        tmp_q, tmp_r = X.shape
        prediction = np.matmul(w2, np.transpose(a1)) + b2*np.ones((tmp_q, 1))
        return prediction[0][0]

    return func


def radbas(M, spread):
    return np.exp(-(M / spread) ** 2 * np.log(2))


def find_large_row(M):
    tmp = np.nan_to_num(M)
    tmp = np.sum(tmp ** 2, axis=1)
    s = list(tmp)
    return s.index(max(s))
