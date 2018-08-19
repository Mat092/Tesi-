#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from random import randint

class RNet_population:
    
    def __init__(self):
        self.medium_fitness = 0
        self.num_individuals = 100
        self.RNpopulation = []
        for n in range(self.num_individuals):
            self.RNpopulation.append(Random_Network())
    
    def pop_fitness(self, X_train, X_test, y_train, y_test): #calcola il fitness
        for RandNet in self.RNpopulation:
            RandNet.RNfitness(X_train, X_test, y_train, y_test)
    
    def medium_pop_fitness(self): #calcola il fitness medio della popolazione
        fitness_tot = 0
        for RandNet in self.RNpopulation:
            fitness_tot = fitness_tot + RandNet.fitness
        self.medium_fitness = fitness_tot / self.num_individuals
       
    def Bubble_sort(self): #slow
        sort = Random_Network()
        for ind1 in range(len(self.RNpopulation)-1):
            for ind2 in range(ind1,len(self.RNpopulation)-1,1):
                if self.RNpopulation[ind1].fitness > self.RNpopulation[ind2].fitness:
                    sort = self.RNpopulation[ind1]
                    self.RNpopulation[ind2] = self.RNpopulation[ind1]
                    self.RNpopulation[ind1] = sort
    
    def shortBubbleSort(self):
        exchanges = True
        passnum = len(self.RNpopulation)-1
        while passnum > 0 and exchanges:
           exchanges = False
           for i in range(passnum):
               if self.RNpopulation[i].fitness < self.RNpopulation[i+1].fitness:
                   exchanges = True
                   temp = self.RNpopulation[i]
                   self.RNpopulation[i] = self.RNpopulation[i+1]
                   self.RNpopulation[i+1] = temp
           passnum = passnum-1
        
    def Print_Best(self): #After Bubble_sort
        print("mean fitness: ",self.medium_fitness)
        self.RNpopulation[0].Print_RNet()
        
    def Best_Score(self):       #after Bubble sort
        return self.RNpopulation[0].fitness
    
    def Print_pop(self):
        for ind in self.RNpopulation:
            ind.Print_RNet()
            
        
        
class Random_Network:
    
    def __init__(self):
        self.fitness = 0
        self.genome = []
        self.num_hidden_layers = randint(1,10)
        for n in range(self.num_hidden_layers):
            self.genome.append(randint(1,100))
            
    def RNfitness(self, X_train, X_test, y_train, y_test): 
        tpl = tuple(self.genome)
        net = MLPClassifier(hidden_layer_sizes = tpl,
                                          alpha = 1, activation = 'relu')
        net.fit(X_train, y_train)
        self.fitness = net.score(X_test, y_test)
        
    def Crossover1(self, RandNet): #sommo i contributi tagliati in x e y 
        son1 = []                  #dalle due network
        son2 = []
        x = randint(0,len(self.genome)-1)
        y = randint(0,len(RandNet.genome)-1)
        son1 = self.genome[:x] + RandNet.genome[y:]
        son2 = self.genome[x:] + RandNet.genome[:y]
        return son1,son2
            
    def Mutation(self):
        #mutazione, può anche allungare o accorciare il genoma di 1 carattere
        x = randint(0,100)
        y = randint(0,100)
        if x < 30:
            if x < 15:
                self.genome.append(randint(1,1000))
            elif len(self.genome) != 1:
                z = randint(0,self.num_hidden_layers - 1)
                del self.genome[z]
        if y < 30:
            z = randint(0,len(self.genome) - 1)
            self.genome[z] = randint(1,1000)
        
    def Print_RNet(self):
        print("Hidden_Layer:",self.genome,sep = "\t")
        print("Fitness:",self.fitness,sep = "\t")
        
        
        
        
        
        
        
        
        
        
        