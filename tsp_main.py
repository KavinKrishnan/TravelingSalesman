import sys
from TSPBranchBoundEdge import branchBound
from TSP_LS_SA import simAnneal
from TSPHeuristic import mstApprox
from TSPLocalGA import genetic_algorithm
import random as rand
import numpy as np
import time

file_name = sys.argv[2]
algorithm = sys.argv[4]
cut_time = float(sys.argv[6])
seed = 10
if len(sys.argv) > 8:
    seed = float(sys.argv[8])

def backup():
    start = time.time()
    rand.seed(seed)
    dataFile = open(file_name, 'r')
    Name = dataFile.readline().strip().split()[1]
    Comment = dataFile.readline().strip().split()[1]
    Dimension = dataFile.readline().strip().split()[1]
    EdgeWeightType = dataFile.readline().strip().split()[1]
    dataFile.readline()
    points = []
    N = int(Dimension)
    for i in range(N):
        x, y = dataFile.readline().strip().split()[1:]
        points.append([float(x), float(y)])
    dataFile.close()
    WEIGHTS = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            currWeight = float('inf')
            if i != j:
                # Weight from Euclidian L2-Norm
                a = np.array(points[i])
                b = np.array(points[j])
                currWeight = np.linalg.norm(a - b)
            WEIGHTS[i, j] = currWeight
    WEIGHT_ORIGINAL = np.copy(WEIGHTS)
    initSolution = [i for i in range(N)]
    rand.shuffle(initSolution)
    initCost = 0
    lastNode = initSolution[-1]
    for c in initSolution:
        initCost += WEIGHT_ORIGINAL[lastNode, c]
        lastNode = c
    totalTime = time.time() - start
    return initSolution, initCost, totalTime

solution = None
cost = None
runtime = None
if algorithm == "BnB":
    try:
        solution, cost, runtime = branchBound(file_name, cut_time, seed)
    except:
        solution, cost, runtime = backup()
if algorithm == "Approx":
    try:
        solution, cost, runtime = mstApprox(file_name, cut_time, seed)
    except:
        solution, cost, runtime = backup()
if algorithm == "LS1":
    try:
        solution, cost, runtime = simAnneal(file_name, cut_time, seed)
    except:
        solution, cost, runtime = backup()
if algorithm == "LS2":
    try:
        solution, cost, runtime = genetic_algorithm(file_name, cut_time, seed)
    except:
        solution, cost, runtime = backup()

city = file_name.split("/")[1]
city_name = city.split(".")[0]
if algorithm == "BnB":
    entire = city_name + "_" + algorithm + "_" + str(cut_time) + ".sol"
else:
    entire = city_name + "_" + algorithm + "_" + str(cut_time) + "_" + str(seed) + ".sol"
outputfile = open("output/" + entire, "w")
sol = ''
for s in solution:
    sol += str(s) + ','
sol = sol[:len(sol) - 1]
outputfile.write(str(cost) + '\n' + sol)
outputfile.close()

if algorithm == "BnB":
    entire = city_name + "_" + algorithm + "_" + str(cut_time) + ".trace"
else:
    entire = city_name + "_" + algorithm + "_" + str(cut_time) + "_" + str(seed) + ".trace"
outputfile = open("output/" + entire, "a")
tme = str(round(runtime, 2))
outputfile.write(str(tme) + ', ' + str(int(cost)) + '\n')
outputfile.close()
