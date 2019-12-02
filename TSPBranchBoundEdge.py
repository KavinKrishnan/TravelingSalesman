import numpy as np
from queue import PriorityQueue
from itertools import count
import math
import sys
import random as rand
import time

def branchBound(file,cutTime,rseed):

    rand.seed(rseed)
    start = time.time()

    # 1. Parses .tsp file of a City and constructs array "points" where points[i] represents the coordinates of data point with ID #i
    dataFile = open(file, 'r')

    Name = dataFile.readline().strip().split()[1]
    Comment = dataFile.readline().strip().split()[1]
    Dimension = dataFile.readline().strip().split()[1]
    EdgeWeightType = dataFile.readline().strip().split()[1]
    dataFile.readline()

    points = []

    N = int(Dimension)
    for i in range(N):
        x,y = dataFile.readline().strip().split()[1:]
        points.append([float(x), float(y)])

    dataFile.close()

    # 2. Creates edge weight matrix WEIGHTS where WEIGHTS[i, j] represents weight between data point ID#i and data point ID#j:
    # if i==j, meaning the the same data point and itself, will have a distance of float('inf') which is infinity

    WEIGHTS = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            currWeight = float('inf')
            if i != j:
                # Weight from Euclidian L2-Norm
                a = np.array(points[i])
                b = np.array(points[j])
                currWeight = np.linalg.norm(a-b)
            WEIGHTS[i,j] = currWeight

    WEIGHT_ORIGINAL = np.copy(WEIGHTS)

    # 3. Reduce Initial Matrix of Weights (adjancy matrix)
    def matrixReduce(weights):
        neededReduce = False
        reduceCost = 0

        for i in range(N):
            minElement = np.min(weights[i,:])
            if (minElement != 0) and (minElement < float('inf')):
                neededReduce = True
                reduceCost += minElement
                weights[i, :] -= minElement

        for i in range(N):
            minElement = np.min(weights[:,i])
            if (minElement != 0) and (minElement < float('inf')):
                neededReduce = True
                reduceCost += minElement
                weights[:,i] -= minElement

        return (weights, reduceCost)

    # Reduce weights and get costs
    WEIGHTS, reduceCost = matrixReduce(WEIGHTS)

    # 4. Initialize Root of State Tree, UpperBound, PriorityQueue "nodes" to expand
    nodes = PriorityQueue()
    root = (reduceCost, [], WEIGHTS)
    nodes.put(root)
    UPPERBOUND = float('inf')
    bestSol = [i for i in range(N)]
    rand.shuffle(bestSol)


    while not nodes.empty():
        if (time.time() - start) >= cutTime:
            if len(bestSol) > 0:
                bestSol += [(bestSol[-1][1], bestSol[0][0])]

            cost = 0
            for (r, c) in bestSol:
                cost += WEIGHT_ORIGINAL[r, c]

            sol = []
            for i in range(len(bestSol)):
                sol.append(bestSol[i][0])

            return sol, cost

        curr = nodes.get()
        currCost, currSol, currMatrix = curr

        if currCost >= UPPERBOUND:
            continue

        if len(currSol) > 0:
            # Get last visited vertex
            splitRow = currSol[-1][1]
            splitCol = np.argsort(currMatrix[splitRow, :])[0]
        else:
            # Finds the biggest difference between first and second smallest edges from a vertex
            sortedCurrM = np.sort(currMatrix, axis=1)
            sortedCurrM[sortedCurrM == float('inf')] = sys.float_info.max
            diffCurrM = sortedCurrM[:, 1] - sortedCurrM[:, 0]
            diffCurrM[np.isnan(diffCurrM)] = -1
            diffCurrM = np.argsort(diffCurrM)

            # Selects to split on the smallest edge from that vertex
            splitRow = diffCurrM[-1]
            splitCol = np.argsort(currMatrix[splitRow, :])[0]

        if currMatrix[splitRow, splitCol] is float('inf'):
            if currCost < UPPERBOUND:
                UPPERBOUND = currCost
                bestSol = currSol
            continue

        if len(currSol) == (N-1):
            row, col = np.unravel_index(currMatrix.argmin(), currMatrix.shape)
            currSol += [(row,col)]
            if currCost < UPPERBOUND:
                UPPERBOUND = currCost
                bestSol = currSol
            continue

        if len(currSol) == (N-2):
            row, col = np.unravel_index(currMatrix.argmin(), currMatrix.shape)
            currSol += [(row,col)]
            if currCost < UPPERBOUND:
                UPPERBOUND = currCost
                bestSol = currSol
            continue

        # Left child representing solutions with Selected Edge
        leftMatrix = np.copy(currMatrix)
        leftMatrix[splitRow,:] = float('inf')
        leftMatrix[:,splitCol] = float('inf')
        leftMatrix[:, splitRow] = float('inf')
        leftMatrix[splitCol, splitRow] = float('inf')
        leftMatrix, leftReduce = matrixReduce(leftMatrix)
        leftSol = currSol + [(splitRow,splitCol)]
        leftCost = currCost + leftReduce #Update Lower Bound with reduction cost
        nodes.put((leftCost, leftSol, leftMatrix))

        #Right child representing solutions without Selected Edge
        rigtMatrix = np.copy(currMatrix)
        rigtMatrix[splitRow, splitCol] = float('inf')
        rigtMatrix[splitCol, splitRow] = float('inf')
        rigtMatrix, rightReduce = matrixReduce(rigtMatrix)
        rightSol = currSol
        rightCost = currCost + rightReduce #Update Lower Bound with reduction cost
        nodes.put((rightCost, rightSol, rigtMatrix))


    bestSol += [(bestSol[-1][1], bestSol[0][0])]

    cost = 0
    for (r,c) in bestSol:
        cost += WEIGHT_ORIGINAL[r,c]

    sol = []
    for i in range(len(bestSol)):
        sol.append(bestSol[i][0])

    return sol, cost

# answer = branchBound('./DATA/Atlanta.tsp',1,0)
# print(answer[0])
# print(answer[1])





