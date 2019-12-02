import numpy as np
from queue import Queue
from itertools import count
import math
import sys
import time
import random as rand

# This file is the Traveling Salesman solution for heuristic; We've used MST 2-Approx to solve it

def mstApprox(file, cutTime, rseed):
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
        initCost += WEIGHT_ORIGINAL[lastNode, c]
        lastNode = c

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

    # Define minimum key used for Prim's Algorithm to get MST

    def minimum(node, tree):
        min = sys.maxsize
        for v in range(len(points)):
            if node[v] < min and not tree[v]:
                min = node[v]
                result = v
        return result

    # 3. Run Prim's Algorithm to get MST

    key = [sys.maxsize] * len(points)
    parent = [None] * len(points)
    key[0] = 0
    mst = [False] * len(points)
    parent[0] = -1
    for count in range(len(points)):
        minkey = minimum(key, mst)
        mst[minkey] = True
        for v in range(len(points)):
            if (time.time() - start) >= cutTime:
                return initSolution, initCost
            if WEIGHTS[minkey][v] > 0:
                if not mst[v]:
                    if key[v] > WEIGHTS[minkey][v]:
                        key[v] = WEIGHTS[minkey][v]
                        parent[v] = minkey
    msTree = []
    for i in range(1, len(points)):
        msTree.append((parent[i], i))

    # Define Node class to represent tree structure

    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    # 4. Do a pre-order listing of the MST to get the cities traveled

    def treePreorder(root):
        order = []
        if root is None:
            return
        nStack = []
        nStack.append(root)
        while(len(nStack) > 0):
            curr = nStack.pop()
            order.append(curr.val)
            if curr.right is not None:
                nStack.append(curr.right)
            if curr.left is not None:
                nStack.append(curr.left)
        return order

    # Put MST solution (tuples) into tree format

    tree = {}
    for tup in msTree:
        if tup[0] in tree:
            tree[tup[0]]["right"] = tup[1]
        else:
            tree[tup[0]] = {"left": tup[1], "right": None}

    root = Node(0)
    nextNodes = Queue()
    nextNodes.put(root)
    while not nextNodes.empty():
        if (time.time() - start) >= cutTime:
            return initSolution, initCost
        currNode = nextNodes.get()
        val = currNode.val
        if val in tree:
            if tree[val]["left"] is not None:
                currNode.left = Node(tree[val]["left"])
                nextNodes.put(currNode.left)
            if tree[val]["right"] is not None:
                currNode.right = Node(tree[val]["right"])
                nextNodes.put(currNode.right)

    # Build solution and get cost

    tsp_solution = treePreorder(root)
    tsp_solution.append(0)
    cost = 0
    last = tsp_solution[0]

    for i in range(1, len(tsp_solution)):
        a = np.array(points[last])
        b = np.array(points[tsp_solution[i]])
        cost += np.linalg.norm(a - b)
        last = tsp_solution[i]
    return tsp_solution, cost, time.time() - start
