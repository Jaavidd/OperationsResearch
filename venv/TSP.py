# # -*- coding: utf-8 -*-
# """
# Solves a random TSP-instance up to approx 60 points.
# """
#
import cvxpy as cy
import numpy as np
import matplotlib.pyplot as plt


def plot_solution(points, lines):
    """
    Plots the tour.

    points:  an nx2 array of floats
    lines: an nxn array:   lines[i,j]=1 means a line from i to j.
    """
    n = len(points)

    x = [];
    y = []
    for i in range(n):
        x.append(points[i][0])
        y.append(points[i][1])

        plt.plot(x, y, 'co', color='r', markersize=2)
    plt.plot(x[0], y[0], 'co', color='b')

    a_scale = float(max(x)) / float(100)

    for i in range(n):
        for j in range(n):
            if lines[i, j] == 1:
                plt.arrow(x[i], y[i], (x[j] - x[i]), (y[j] - y[i]),
                          head_width=a_scale, linewidth=0.5,
                          color='g', length_includes_head=True)

    # Set axis too slitghtly larger than the set of x and y
    plt.xlim(min(x) - 0.05 * (max(x) - min(x)), max(x) + 0.05 * (max(x) - min(x)))
    plt.ylim(min(y) - 0.05 * (max(y) - min(y)), max(y) + 0.05 * (max(y) - min(y)))
    plt.show()


def compute_dist(a, b):
    """
    computes the euclidean distance between two points in R2
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def permutation(xval, n):
    """
    input: the solution x
    output: an array with j on position i if there is an arc (i,j) in solution x

    Example: If the solution contains  the subtours  0-1-0 and 2-3-4-2 then the
    output is :  [1,0,3,4,2]
    """
    y = np.dot(xval, np.arange(n))  # the matrix product of xval and the array [0,1,2,...,n-1]
    return [int(j) for j in y]  # since we use these values as indices of arrays we need them to be integer


def components(perm):
    """
    returns :
        comps: the components of the permutation as a list of lists
        subtours: True if there are subtours

    Example: If perm = [1,0,3,4,2] then it returns
        [[0,1][2,3,4]], True
    """
    comps = []  # no components
    rest = set(perm)  # the remaining points as a set
    while rest != set([]):
        v0 = min(rest)  # select a point in the remaining set of points
        list = [v0]
        next = perm[v0]
        while next != v0:
            list += [next]
            next = perm[next]
        comps += [list]
        rest -= set(list)  # delete the component from the remaining vertices
    subtours = len(comps) > 1
    return comps, subtours


def main():
    n = 30  # number of points
    np.random.seed(1)
    points = np.random.random(size=(n, 2))  # create random points

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            D[i, j] = compute_dist(points[i], points[j])
            D[j, i] = D[i, j]

    x = cy.Variable((n,n),boolean=True)  # x[i,j]=1 if there is an edge from i to j

    L = cy.Variable()  # the length of the tour or total length of the subtours

    ## the constraints
    constraints = [x >= 0]  # a dummy inequality. GLPK_MI fails if there are no inequalities

    # TO DO (1): Add the contraint that for each point i exactly one arc is leaving i
    # for c in comps:

    # TO DO (2): Add the contraint that for each point i exactly one arc is enterng  i

    # TO DO (3): Add the contraint that there are no loops
    constraints=[ x[i,i]==0 ]
    constraints += [L == sum(x[i, j] * D[i, j] for i in range(n) for j in range(n))]
    # L is the length of the tour

    objective = cy.Minimize(L)
    prob = cy.Problem(objective, constraints)

    optimal = False
    n_iterations = 1

    while optimal == False:
        prob.solve(solver=cy.GLPK_MI)  ## solving the problem

        xval = np.array(x.value)

        perm = permutation(xval, n)  # wrtite the solution as a permutation
        plot_solution(points, xval)

        print("iteration: %s" % n_iterations)

        comps, subtours = components(perm)  #

        if subtours:
            print("%s SUBTOURS!\n" % len(comps))

            # TO DO (4):  Remove  the next line of code (optimal = True) and add the subtour elimination constraint
            optimal = True


        else:
            optimal = True

        n_iterations += 1

    print("OPTIMAL! Length is %s" % L.value)


# main()
#
#


# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:58:36 2019

@author: Andrews-mobile
"""
import pandas as pd
import numpy as np


def readWithpd():
    data = pd.read_excel('weather-debilt.xlsx')

    np.matrix
    return data


def convertValues(data):
    for d in data:
        if d[2] <= 2:
            d[2] = 0
        elif d[2] > 2 and d[2] <= 5:
            d[2] = 1
        elif d[2] > 5 and d[2] <= 7:
            d[2] = 2
        else:
            d[2] = 3

        if d[1] <= 20:
            d[1] = 0
        else:
            d[1] = 1


def computeTrCloudFractions(data, CStCount):
    P = np.zeros(16)
    for i in range(len(data)):
        if i > 0:
            P[4 * data[i - 1][2] + data[i][2]] += 1
    for i in range(len(P)):
        # print(p)
        P[i] = P[i] / CStCount[i // 4]
    return P


def cloudStateCount(data):
    V = np.zeros(4)
    for d in data:
        V[d[2]] += 1
    return V


def emissionProbabilities(data, CStCount):
    EP = np.zeros(8)
    for d in data:
        EP[2 * d[2] + d[1]] += 1
    for i in range(len(EP)):
        EP[i] = EP[i] / CStCount[i // 2]
    return EP


def arrangePrMatrix(P):
    M = []
    m = []
    for i in range(len(P)):
        m.append(P[i])
        if (i + 1) % 4 == 0 and i != 0:
            M.append(m)
            m = []
    M[2][3] = 1 - M[2][0] - M[2][1] - M[2][2]  # fixing the row to sum up to exactly 1
    M = np.matrix(M)
    # print()
    # print(M)
    return M
    # for m in M:
    #    print(np.sum(m))


def arrangePrMat(P):
    M = []
    m = []
    for i in range(len(P)):
        m.append(P[i])
        if (i + 1) % 4 == 0 and i != 0:
            M.append(m)
            m = []
    # print()
    # print(M)
    return M
    # for m in M:
    #    print(np.sum(m))


def arrangeEmMatrix(P):
    M = []
    m = []
    for i in range(len(P)):
        m.append(P[i])
        if (i + 1) % 2 == 0 and i != 0:
            M.append(m)
            m = []

    return M


def matrixMultiply(M):
    for i in range(8):
        M = np.dot(M, M)
    return M


def printProbs(P):

    print('%8d' % 0, end=' ')
    print('%8d' % 1, end=' ')
    print('%8d' % 2, end=' ')
    print('%8d' % 3, end=' ')
    for i in range(len(P)):
        if (i % 4 == 0):
            print()
            print('%d' % (i / 4), end='   ')
        print('%8.6f' % P[i], end=' ')


def printEmProb(EmiProb):

    print('%8d' % 0, end=' ')
    print('%8d' % 1, end=' ')
    for i in range(len(EmiProb)):
        if (i % 2 == 0):
            print()
            print('%d' % (i / 2), end='   ')
        print('%8.6f' % EmiProb[i], end=' ')


def HMMdecoding(y, S, a, PTr, EmiProb):
    n = len(y) - 1
    M = len(S)
    beta = np.zeros([M, n + 1])

    for i in S:
        beta[i, -1] = EmiProb[i][y[-1]]
    for t in range(n - 1, -1, -1):
        for i in S:
            beta[i, t] = EmiProb[i][y[t]] * max(PTr[i][:] * beta[:, t + 1])

    x = [-1] * (n + 1)  # declares a vector of n+1 integers (-1)
    x[0] = np.argmax(a * beta[:, 0])
    for t in range(1, n + 1):
        x[t] = np.argmax(PTr[x[t - 1]   ][:] * beta[:, t])
    return x


def main():
    Data = readWithpd().as_matrix()
    convertValues(Data)
    CStCount = cloudStateCount(D)
    PTr = computeTrCloudFractions(Data, CStCount)
    EmiProb = emissionProbabilities(Data, CStCount)
    printProbs(PTr)
    print('\n')
    printEmProb(EmiProb)
    print('\n')
    M = arrangePrMatrix(PTr)
    Dist = matrixMultiply(M)
    print()
    print('DISTRIBUTION = ', end='')
    print(Dist.item(0), Dist.item(1), Dist.item(2), Dist.item(3), sep=' , ')
   # print()
    S = [0, 1, 2, 3]
    y = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    EmiProb = arrangeEmMatrix(EmiProb)
    PTr = arrangePrMat(PTr)

    X = HMMdecoding(y, S, [Dist.item(0), Dist.item(1), Dist.item(2), Dist.item(3)], PTr, EmiProb)
    print('MAXIMUM PROBABILITY PATH FOR GIVEN OUTPUT')
    print(X)


if __name__ == '__main__':
    main()