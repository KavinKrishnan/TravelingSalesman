import numpy as np
from queue import PriorityQueue
from itertools import count
import math
import sys
import random as rand
import time

def simAnneal(file,cutTime,rseed):

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

    # 3. Create initial solution at random and get initial cost
    initSolution = [i for i in range(N)]
    rand.shuffle(initSolution)

    initCost = 0
    lastNode = initSolution[-1]
    for c in initSolution:
        initCost += WEIGHT_ORIGINAL[lastNode,c]
        lastNode = c

    # 4. Simulated Annealing Step - anneal until temperature hits terminal temperature
    currTemp = 1
    terminalTemp = 0.0000001
    coolRate = 0.95
    currCost = initCost
    currSolution = initSolution
    coolCount = 0

    while currTemp > terminalTemp:

        if (time.time() - start) >= cutTime:
            return currSolution, currCost, time.time() - start

        newSolution = currSolution
        swapNode0Ind, swapNode1Ind = rand.sample(range(N),2)

        swapNode0 = currSolution[swapNode0Ind]
        swapNode1 = currSolution[swapNode1Ind]

        newSolution[swapNode0Ind] = swapNode1
        newSolution[swapNode1Ind] = swapNode0

        newCost = 0
        lastNode = newSolution[-1]
        for c in newSolution:
            newCost += WEIGHT_ORIGINAL[lastNode, c]
            lastNode = c

        if newCost < currCost:
            currSolution = newSolution
            currCost = newCost
        elif np.exp((currCost - newCost) / currTemp) > rand.random():
            currSolution = newSolution
            currCost = newCost

        if coolCount == ((N+1)*N):
            currTemp *= coolRate
            coolCount = 0
        # currTemp *= coolRate

        coolCount += 1

    return currSolution, currCost, time.time() - start

# answer = simAnneal('./DATA/Boston.tsp',1,0)
# print(answer[0])
# print(answer[1])












































# initSolutionWeight = np.copy(WEIGHTS)
# initSolution = [0]
# initSolutionWeight[:,0] = float('inf')
# lastNode = 0
# for c in range(1,N):
#     nextNode = np.argmin(initSolutionWeight[lastNode])
#     initSolutionWeight[:, nextNode] = float('inf')
#     initSolution.append(nextNode)
#     lastNode = nextNode



    # # Calculate Cost of New Potential Solution
    # # swapNode0 curr Edges
    # subEdge0 = 0
    # subEdge1 = 0
    #
    # # swapNode1 curr Edges
    # subEdge2 = 0
    # subEdge3 = 0
    #
    # if swapNode0Ind == 0:
    #     subEdge0 = WEIGHT_ORIGINAL[currSolution[-1],swapNode0]
    #     subEdge1 = WEIGHT_ORIGINAL[swapNode0,currSolution[1]]
    # if swapNode0Ind == N-1:
    #     subEdge0 = WEIGHT_ORIGINAL[currSolution[swapNode0Ind-1], swapNode0]
    #     subEdge1 = WEIGHT_ORIGINAL[swapNode0, currSolution[0]]
    # else:
    #     subEdge0 = WEIGHT_ORIGINAL[currSolution[swapNode0Ind-1], swapNode0]
    #     subEdge1 = WEIGHT_ORIGINAL[swapNode0, currSolution[swapNode0Ind+1]]
    #
    # if swapNode1Ind == 0:
    #     subEdge2 = WEIGHT_ORIGINAL[currSolution[-1],swapNode1]
    #     subEdge3 = WEIGHT_ORIGINAL[swapNode1,currSolution[1]]
    # if swapNode1Ind == N-1:
    #     subEdge2 = WEIGHT_ORIGINAL[currSolution[N-2], swapNode1]
    #     subEdge3 = WEIGHT_ORIGINAL[swapNode1, currSolution[0]]
    # else:
    #     subEdge2 = WEIGHT_ORIGINAL[currSolution[swapNode1Ind-1], swapNode1]
    #     subEdge3 = WEIGHT_ORIGINAL[swapNode1, currSolution[swapNode1Ind+1]]
    #
    # # swapNode0 new Edges
    # addEdge0 = 0
    # addEdge1 = 0
    #
    # # swapNode1 curr Edges
    # addEdge2 = 0
    # addEdge3 = 0
    #
    # if swapNode1Ind == 0:
    #     addEdge0 = WEIGHT_ORIGINAL[currSolution[-1],swapNode0]
    #     addEdge1 = WEIGHT_ORIGINAL[swapNode0,currSolution[1]]
    # if swapNode1Ind == N-1:
    #     addEdge0 = WEIGHT_ORIGINAL[currSolution[N-2], swapNode0]
    #     addEdge1 = WEIGHT_ORIGINAL[swapNode0, currSolution[0]]
    # else:
    #     addEdge0 = WEIGHT_ORIGINAL[currSolution[swapNode1Ind-1], swapNode0]
    #     addEdge1 = WEIGHT_ORIGINAL[swapNode0, currSolution[swapNode1Ind+1]]


















