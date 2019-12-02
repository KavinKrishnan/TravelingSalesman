import numpy as np
from queue import Queue
from itertools import count
import math
import sys
import random as rand
import operator
import pandas as pd
import time

# This file is the Traveling Salesman solution for local search using Genetic Algorithms

def genetic_algorithm(file, cutTime, rseed):
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

    # 3. Create initial solution at random and get initial cost
    initSolution = [i for i in range(N)]
    rand.shuffle(initSolution)

    initCost = 0
    lastNode = initSolution[-1]
    for c in initSolution:
        initCost += WEIGHT_ORIGINAL[lastNode, c]
        lastNode = c

    # Create fitness class for genetic algorithm where fitness is inverse of weight
    class Fitness:
        def __init__(self, individual):
            self.individual = individual
            self.distance = 0
            self.fitness = 0.0

        def individualWeight(self):
            if self.distance == 0:
                dist = 0
                for i in range(len(self.individual)):
                    src = self.individual[i]
                    if i + 1 < len(self.individual):
                        dest = self.individual[i + 1]
                    else:
                        dest = self.individual[0]
                    dist += WEIGHTS[src][dest]
                self.distance = dist
            return self.distance

        def fitnessFunction(self):
            if self.fitness == 0:
                self.fitness = 1 /float(self.individualWeight())
            return self.fitness

    # Genetic algorithm functions to get a population of routes
    def getIndividual(locs):
        return rand.sample([i for i in range(len(locs))], len(locs))

    def populationMethod(sz, locs):
        population = []
        for i in range(sz):
            population.append(getIndividual(locs))
        return population

    def rank(population):
        res = {}
        for i in range(len(population)):
            res[i] = 1 / Fitness(population[i]).fitnessFunction()
        return sorted(res.items(), key=operator.itemgetter(1), reverse=True)

    # Choose the best individuals/routes
    def selection(ranked, newSize):
        res = []
        df = pd.DataFrame(np.array(ranked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
        for i in range(newSize):
            res.append(ranked[i][0])
        for i in range(len(ranked) - newSize):
            pick = 100 * rand.random()
            for i in range(len(ranked)):
                if pick <= df.iat[i, 3]:
                    res.append(ranked[i][0])
                    break
        return res

    def pool(population, res):
        matingPool = []
        for i in range(0, len(res)):
            index = res[i]
            matingPool.append(population[index])
        return matingPool

    def breed(p1, p2):
        c1 = []
        gA = int(rand.random() * len(p1))
        gB = int(rand.random() * len(p1))
        start = min(gA, gB)
        end = max(gA, gB)
        for i in range(start, end):
            c1.append(p1[i])
        c2 = [i for i in p2 if i not in c1]
        c = c1 + c2
        return c

    def breedPop(mpool, sz):
        children = []
        length = len(mpool) - sz
        pool = rand.sample(mpool, len(mpool))
        for i in range(sz):
            children.append(mpool[i])
        for i in range(length):
            children.append(breed(pool[i], mpool[len(mpool) - i - 1]))
        return children

    def mutate(individual, rate):
        for ind in range(len(individual)):
            if (rand.random() < rate):
                indNew = int(rand.random() * len(individual))
                c1 = individual[ind]
                c2 = individual[indNew]
                individual[ind] = c2
                individual[indNew] = c1
        return individual

    def mutatePopulation(population, rate):
        pop = []
        for i in range(len(population)):
            pop.append(mutate(population[i], rate))
        return pop

    def nextGeneration(current, sz, rate):
        ranked = rank(current)
        res = selection(ranked, sz)
        matingPool = pool(current, res)
        children = breedPop(matingPool, sz)
        nextGen = mutatePopulation(children, rate)
        return nextGen

    def GA(population, size, newSize, rate, generations):
        pop = populationMethod(size, population)
        for i in range(generations):
            if (time.time() - start) >= cutTime:
                return initSolution, initCost
            pop = nextGeneration(pop, newSize, rate)
        optimalIdx = rank(pop)[0][0]
        return pop[optimalIdx]

    opt = GA(population=points, size=100, newSize=20, rate=0.01, generations=500)
    if len(opt) == 2:
        return opt
    opt.append(opt[0])
    cost = 0
    last = opt[0]

    for i in range(1, len(opt)):
        a = np.array(points[last])
        b = np.array(points[opt[i]])
        cost += np.linalg.norm(a - b)
        last = opt[i]
    return opt, cost, time.time() - start
