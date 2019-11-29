import numpy as np
from queue import PriorityQueue
from itertools import count

# 1. Parses .tsp file of a City and constructs array "points" where points[i] represents the coordinates of data point with ID #i
dataFile = open('./DATA/Atlanta.tsp', 'r')

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


# 3. Reduce Initial Matrix of Weights
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

WEIGHTS, reduceCost = matrixReduce(WEIGHTS)


# 4. Initialize root of state tree, upperbound "UPPERBOUND", and priority queue "nodes" for Least Cost B & B

startID = 0

UPPERBOUND = float('inf')
OPTIMAL = []
nodes = PriorityQueue()
startWEIGHTS = np.copy(WEIGHTS)
startWEIGHTS[:, startID] = float('inf')
tiebreaker = count()
root = (reduceCost, startID, next(tiebreaker), startWEIGHTS, [startID])
nodes.put(root)

while not nodes.empty():
    nextExpand = nodes.get()
    currCost, currID, currTie, currMatrix, currSolution = nextExpand
    # print(currCost, currID, currSolution)

    if currCost > UPPERBOUND:
        break

    nextNodes = []

    for j in range(N):
        currElement = currMatrix[currID, j]
        if currElement < float('inf'):
            newID = j
            newMatrix = np.copy(currMatrix)
            newMatrix[currID, :] = float('inf')
            newMatrix[:, j] = float('inf')
            newMatrix[j,currID] = float('inf')
            newMatrix, reduceCost = matrixReduce(newMatrix)
            newCost = reduceCost + WEIGHTS[currID, j] + currCost
            newSolution = currSolution + [j]
            nextNodes.append((newCost, newID, newMatrix, newSolution))

    if len(nextNodes) == 0:
        if currCost < UPPERBOUND:
            UPPERBOUND = currCost
            OPTIMAL = currSolution + [startID]
    else:
        for n in nextNodes:
            newCost, newID, newMatrix, newSolution = n
            if newCost < UPPERBOUND:
                nodes.put((newCost, newID, next(tiebreaker), newMatrix, newSolution))


cost = 0
solution = OPTIMAL
last = solution[0]

for i in range(1, len(solution)):
    a = np.array(points[last])
    b = np.array(points[solution[i]])
    cost += np.linalg.norm(a - b)
    last = solution[i]

print(cost)
print(solution)












