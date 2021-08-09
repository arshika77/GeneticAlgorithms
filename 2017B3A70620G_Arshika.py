import numpy as np 
from numpy.random import choice
from numpy.random import randint
import random 
from matplotlib import pyplot as plt
np.random.seed(77)
random.seed(77)

#-------------------------------------------Functions for the 8 Queens Problem--------------------------------------------------------------------------

def random_individual():
    return [ random.randint(1,8) for _ in range(8) ]

def intial_individual():
    return [ 1 for _ in range(8) ]

def initial_population(popSize=20):
    return [ intial_individual() for _ in range(popSize)]

def fitness(individual):
    clashes = 0
    #count_set = set(individual)
    #horizontal_collisions = abs(8-len(count_set))
    horizontal_collisions = sum([individual.count(queen)-1 for queen in individual])/2
    clashes += horizontal_collisions

    for i in range(8):
        for j in range(8):
            if ( i != j):
                dx = abs(i-j)
                dy = abs(individual[i] - individual[j])
                if(dx == dy):
                    clashes += 1

    return 29 - int(clashes)

def probability_fitness(individual):
    return fitness(individual)/29

def random_pick(population, probabilities):
    populationWithProbabilty = zip(population, probabilities)
    total = sum(w for c, w in populationWithProbabilty)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(population, probabilities):
        if upto + w >= r:
            return c
        upto += w
    
def reproduce(x,y): 
    same_genes = []
    for i in range(8):
        if (x[i]==y[i]):
            same_genes.append(i)
    child = random_individual()
    for gene_index in same_genes:
        child[gene_index] = x[gene_index]
    return child

def mutate(x):
    x = random_individual()
    return x

def kill_population(population,probabilities):
    populationWithProbabilty = zip(population,probabilities)
    finalList = list(populationWithProbabilty)
    sortedList = sorted(finalList, key = lambda x: x[1])
    n = int(len(sortedList)/2)
    for i in range(n+1):
        sortedList[i] = sortedList[n-i-1]

    population_list = [sortedList[i][0] for i in range(len(sortedList))]
    probabilities_list = [sortedList[i][1] for i in range(len(sortedList))]

    return population_list,probabilities_list


def genetic_evolution(population,mutation_probability,ind_arr,fitness_arr):
    new_population = []
    child_fitness = []
    probabilities = [probability_fitness(individual) for individual in population]
    population,probabilities = kill_population(population,probabilities)
    for i in range(len(population)):
        x = random_pick(population, probabilities) 
        y = random_pick(population, probabilities)

        child = reproduce(x, y)

        if random.random() < mutation_probability:
            child = mutate(child)

        new_population.append(child)
        child_fitness.append(fitness(child))

    gen_fitness = logging_func(new_population,child_fitness,ind_arr,fitness_arr)
    return new_population,gen_fitness

def logging_func(new_population,child_fitness, ind_arr, fitness_arr):
    ind_arr.append(new_population)
    fitness_arr.append(child_fitness)

    individualsWithFitness = zip(new_population,child_fitness)
    for i,f in individualsWithFitness:
        print("Individual = {},  Fitness = {}".format(str(i), f))

    return max(child_fitness)

def eightQueens():
    ind_arr = []
    fitness_arr = []
    genWithFitness = []

    population = initial_population(300)

    print('Initial Population = {}'.format(str(population)))

    generation = 1

    while not 29 in [fitness(ind) for ind in population]:
        print("=== Generation {} ===".format(generation))
        population,max_gen_fitness = genetic_evolution(population,0.35,ind_arr,fitness_arr)
        genList = [generation,max_gen_fitness]
        genWithFitness.append(genList)
        print("")
        print("Maximum Fitness = {}".format(max_gen_fitness))
        generation += 1
        
    #print(genWithFitness)
    solution_board = []

    print("Solved in Generation {}!".format(generation-1))

    for individual_sol in population:
        if fitness(individual_sol) == 29:
            print("")
            print("One of the solutions: ")
            solution_board = individual_sol
            print("Individual = {},  Fitness = {}".format(str(individual_sol), fitness(individual_sol)))


#-----------------------------------------Class and function for Travelling Salesman Problem-------------------------------------------------------------------------------------------------

class TSPGeneticAlgorithm():

    def __init__(self):
        self.initialisePopulation()
        self.initialiseCityDistance()
        self.currentGen = 0 
        self.populationFitness = self.calculatePopulationFitness(self.population)
        self.fitnessLog = []

    def initialisePopulation(self):
        cityList = np.arange(14)
        cityList = cityList.astype(int)
        self.population = np.array([cityList for _ in range(20)])


    def initialiseCityDistance(self):
        self.cityDist = np.zeros((14,14))
        self.cityDist[0] = np.array((0,10000,10000,10000,10000,10000,0.15,10000,10000,0.2,10000,0.12,10000,10000))
        self.cityDist[1] = np.array((10000,0,10000,10000,10000,10000,10000,0.19,0.4,10000,10000,10000,10000,0.13))
        self.cityDist[2] = np.array((10000,10000,0,0.6,0.22,0.4,10000,10000,0.2,10000,10000,10000,10000,10000))
        self.cityDist[3] = np.array((10000,10000,0.6,0,10000,0.21,10000,10000,10000,10000,0.3,10000,10000,10000))
        self.cityDist[4] = np.array((10000,10000,0.22,10000,0,10000,10000,10000,0.18,10000,10000,10000,10000,10000))
        self.cityDist[5] = np.array((10000,10000,0.4,0.21,10000,0,10000,10000,10000,10000,0.37,0.6,0.26,0.9))
        self.cityDist[6] = np.array((0.15,10000,10000,10000,10000,10000,0,10000,10000,10000,0.55,0.18,10000,10000))
        self.cityDist[7] = np.array((10000,0.19,10000,10000,10000,10000,10000,0,10000,0.56,10000,10000,10000,0.17))
        self.cityDist[8] = np.array((10000,0.4,0.2,10000,0.18,10000,10000,10000,0,10000,10000,10000,10000,0.6))
        self.cityDist[9] = np.array((0.2,10000,10000,10000,10000,10000,10000,0.56,10000,0,10000,0.16,10000,0.5))
        self.cityDist[10] = np.array((10000,10000,10000,0.3,10000,0.37,0.55,10000,10000,10000,0,10000,0.24,10000))
        self.cityDist[11] = np.array((0.12,10000,10000,10000,10000,0.6,0.18,10000,10000,0.16,10000,0,0.4,10000))
        self.cityDist[12] = np.array((10000,10000,10000,10000,10000,0.26,10000,10000,10000,10000,0.24,0.4,0,10000))
        self.cityDist[13] = np.array((10000,0.13,10000,10000,10000,0.9,10000,0.17,0.6,0.5,10000,10000,10000,0))

    
    def calculateFitness(self,route):
        distance = 0
        route = route.astype(int)
        for i in range(0,14):        
            distance += self.cityDist[route[i]][route[(i+1)%14]]
        return 100000/(distance+1)

    def calculatePopulationFitness(self,population):
        routeFitness = np.array([self.calculateFitness(route) for route in population])
        self.optimalFitness = np.max(routeFitness)
        return routeFitness    

    def calculateProbabilities(self,fitnessArr):
        return fitnessArr/fitnessArr.sum()

    def generateRandomRoute(self,population,probabilities):
        idx = choice(np.arange(len(population)),1,p=probabilities)
        return population[idx].reshape(14)
    
    def reproduce(self,route1,route2):
        index1 = randint(0,14)
        index2 = randint(0,15)
        child = np.zeros(14)

        if index2 < index1:
            temp = index2
            index2 = index1
            index1 = temp

        child[index1:index2] = np.array(route1[index1:index2])
        childP1 = []
        for city in route2:
            if city not in route1[index1:index2]:
                childP1.append(city)
        
        child[:index1] = childP1[:index1]
        child[index2:] = childP1[index1:]

        return child

    def mutation(self,child):
        swapped = randint(0,14)
        swapWith = (swapped+1)%14
        temp = child[swapped]
        child[swapped] = child[swapWith]
        child[swapWith] = temp
        return child

    def geneticAlgorithm(self):
        self.populationFitness = self.calculatePopulationFitness(self.population)
        while(self.currentGen<500):
            print("===Generation Number: {} ==== Distance: {} ==========".format(self.currentGen,(100000/self.optimalFitness)-1))
            new_population = []
            probabilities= self.calculateProbabilities(self.populationFitness)
            popSizeNew = int(1.5*len(self.population))
            if popSizeNew > 200:
                popSizeNew = 200

            elitism = int(0.3*popSizeNew)
            idx = np.argsort(self.populationFitness)[-elitism:]
            new_population = self.population[idx].tolist()

            for _ in range(popSizeNew-elitism):
                route1 = self.generateRandomRoute(self.population,probabilities)
                route2 = self.generateRandomRoute(self.population,probabilities)
                child = self.reproduce(route1,route2)

                #mutationRate = 0.3
                #if random.random() < mutationRate:
                #    child = self.mutation(child)

                mutProb = 1
                if self.currentGen<100:
                    mutProb: 1/(self.currentGen+1)
                else:
                    mutProb = 0.005
                mutateProb = choice([True,False],1,p=[mutProb,1-mutProb])
                if mutateProb:
                    child = self.mutation(child)

                new_population.append(child) 

            self.currentGen+=1
            self.population = np.array(new_population)
            self.population = self.population.astype(int)
            self.populationFitness = self.calculatePopulationFitness(self.population)
            self.optimalFitness = self.populationFitness.max()
            self.fitnessLog.append(self.optimalFitness)

        return self.population[np.argmax(self.populationFitness)]

def travellingSalesmanProblem():

    TSP_GA = TSPGeneticAlgorithm()
    TSP_GA_results = TSP_GA.geneticAlgorithm()
    print(TSP_GA_results)
    #data = np.array(TSP_GA.fitnessLog)
    #np.savez("TSPMGA9", data)

if __name__ == "__main__":
    prob = input("Enter Q for the 8 Queens Problem, or T for the Travelling Salesman Problem: ")
    if prob=='Q':
        eightQueens()
    else:
        travellingSalesmanProblem()

