from sklearn.neural_network import MLPClassifier
from random import uniform #,randint
from numpy.random import randint
import numpy as np
from sklearn.metrics import log_loss, accuracy_score

class RNet_population:
    
    def __init__(self,num_individuals = 10):
        self.fitness_sum = 0 
        self.mean_fitness = 0
        self.num_individuals = num_individuals
        self.RNpopulation = []
        for n in range(self.num_individuals):
            self.RNpopulation.append(Random_Network())    
            
    def __getitem__(self,index):
        return self.RNpopulation[index]         

    def pop_Neurons_product(self):
        for ind in range (len(self.RNpopulation)):
            self.RNpopulation[ind].cal_Neurons_product()
    
    def pop_fitness(self, X_train, X_test, y_train, y_test): #calcola il fitness
        for RandNet in self.RNpopulation:
            RandNet.RNfitness(X_train, X_test, y_train, y_test)
            
    def update_mean_fitness(self):
        fitness_tot = 0
        for i in range(len(self.RNpopulation)): 
            fitness_tot = fitness_tot + self.RNpopulation[i].fitness
        self.fitness_sum = fitness_tot
        self.mean_fitness = round(fitness_tot / len(self.RNpopulation),3)
        
    def BubbleSort(self): #se voglio ordinarli anche per numero di neuroni
        self.pop_Neurons_product()
        sort = Random_Network()
        for ind1 in range(len(self.RNpopulation)):
            for ind2 in range(ind1,len(self.RNpopulation),1):
                if self.RNpopulation[ind1].fitness < self.RNpopulation[ind2].fitness:
                    sort = self.RNpopulation[ind1]
                    self.RNpopulation[ind1] = self.RNpopulation[ind2]
                    self.RNpopulation[ind2] = sort
                elif (self.RNpopulation[ind1].fitness == 
                      self.RNpopulation[ind2].fitness and 
                      self.RNpopulation[ind1].Neurons_product > 
                      self.RNpopulation[ind2].Neurons_product):
                    sort = self.RNpopulation[ind1]
                    self.RNpopulation[ind1] = self.RNpopulation[ind2]
                    self.RNpopulation[ind2] = sort
                    
    def shortBubbleSort(self): #ordina la popolazione per fitness, trovato online
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
           passnum = passnum - 1
        
    def Print_Best(self): #After Bubble_sort
        print("mean fitness: ",self.mean_fitness)
        self.RNpopulation[0].Print_RNet()
        
    def Best_Score(self):       #after Bubble sort
        return self.RNpopulation[0].fitness
    
    def Print_pop(self):
        for ind in self.RNpopulation:
            ind.Print_RNet()
            print()
            
class Random_Network:
    
    def __init__(self):
        self.links = 0
        self.max_neurons = 50
        self.fitness = 0
        self.prob = 0
        self.genome = []
        self.Neurons_product = 0
        self.num_hidden_layers = randint(1,10)
        for n in range(self.num_hidden_layers):
            self.genome.append(randint(1,self.max_neurons))
            
    def RNfitness(self, X_train, X_test, y_train, y_test):  
        tpl = tuple(self.genome)
        net = MLPClassifier(hidden_layer_sizes = tpl, 
                            activation = 'relu',solver = 'adam',
                            alpha = 0.001,
                            learning_rate = 'constant' , learning_rate_init = 0.001,
                            #random_state = 1,
                            tol = 0.0001,
                            early_stopping = False , validation_fraction = 0.1)
        net.fit(X_train, y_train)
        
        #log_loss score
        #y_p = net.predict_proba(X_test)
        #score = log_loss(y_test,y_p)
        
        #accuracy score
        y_p = net.predict(X_test)
        score = accuracy_score(y_test,y_p)
        
        #score
        #score = net.score(X_test,y_test)
        self.scorer = net
        self.fitness = score
        self.update_links()            
        
    def Crossover1(self, RandNet): #sommo i contributi tagliati in x e y 
        Crv_ratio = 1.01           #dalle due network
        son1 = Random_Network()                  
        son2 = Random_Network()
        if uniform(0,1) < Crv_ratio:
            x = randint(0,len(self.genome)+1,size = 3)
            y = randint(0,len(RandNet.genome)+1,size = 3)
            x = int(np.median(x))
            y = int(np.median(y))
            son1.genome = self.genome[:x] + RandNet.genome[y:]
            son2.genome = RandNet.genome[:y] + self.genome[x:]
        else :
            son1.genome = self.genome 
            son2.genome = RandNet.genome
        return son1,son2
    
    def Mutation(self):
        #mutazione, può anche allungare o accorciare la rete di 1 layer
        x = randint(0,100)
        y = randint(0,100)
        if x < 5:
            if x < 2.5:
                self.genome.append(randint(1,self.max_neurons))
            elif len(self.genome):
                self.genome.pop()
        if y < 5:
            if len(self.genome) - 1 and len(self.genome):
                z = randint(0,len(self.genome) - 1)
                self.genome[z] = randint(1,self.max_neurons)
            elif len(self.genome) :
                self.genome[0] = randint(1,self.max_neurons)
            else :
                self.genome.append(randint(1,self.max_neurons))
                
    def update_links(self):
        prd = 0
        links = 0
        for i in range(len(self.genome)-1):
            prd = self.genome[i]*self.genome[i+1]
            links = links + prd 
        self.links = links     
            
    def Print_RNet(self):
        print("Hidden_Layer:",self.genome,sep = "\t")
        print("Fitness:",self.fitness,sep = "\t")
        
        
        
        
        
        
        
        
        
        
        