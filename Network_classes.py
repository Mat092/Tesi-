from sklearn.neural_network import MLPClassifier
from random import uniform #,randint
from numpy.random import randint
import numpy as np
from sklearn.metrics import log_loss, accuracy_score

class RNet_population:
    
    def __init__(self,num_individuals = 10):
        self.num_individuals = num_individuals
        self.RNpopulation = []
        for n in range(self.num_individuals):
            self.RNpopulation.append(Random_Network())    
            
    def __getitem__(self,index):
        return self.RNpopulation[index]         
    
    def pop_fitness(self, X_train, X_test, y_train, y_test): #calcola il fitness
        for RandNet in self.RNpopulation:
            RandNet.RNfitness(X_train, X_test, y_train, y_test)
            
    def update_mean_fitness(self):
        fitness_tot = 0
        for i in range(len(self.RNpopulation)): 
            fitness_tot = fitness_tot + self.RNpopulation[i].fitness
        self.fitness_sum = fitness_tot
        self.mean_fitness = round(fitness_tot / len(self.RNpopulation),3)
        self.update_mean_links()
                    
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
    
    def update_mean_links(self):
        tot_links = 0 
        for ind in self.RNpopulation:
            tot_links = tot_links + ind.links
        self.mean_links = tot_links / self.num_individuals 
        
    def Print_Best(self): #After sort
        print("mean fitness: ",self.mean_fitness)
        self.RNpopulation[0].Print_RNet()
        
    def Best_Score(self):       #after sort
        return self.RNpopulation[0].fitness
    
    def Print_pop(self):
        for ind in self.RNpopulation:
            ind.Print_RNet()
            print()
          
class Random_Network:
    
    def __init__(self):
        self.max_neurons = 25
        self.fitness = 0
        self.num_hidden_layers = randint(1,5)
        self.genome = []
        self.Neurons_product = 0
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
        self.update_links(net) 
        
        #log_loss score
        y_p = net.predict_proba(X_test)
        score = log_loss(y_test,y_p)
        
#         accuracy score
#        y_p = net.predict(X_test)
#        score = accuracy_score(y_test,y_p)
        
        #score
#        score = net.score(X_test,y_test)
        self.scorer = net
        self.fitness = score
                   
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
        len_genome = len(self.genome)
        if x < 5:
            if x < 2.5:
                self.genome.append(randint(1,self.max_neurons))
            elif len_genome:
                self.genome.pop()
                
        len_genome = len(self.genome) #update len_genome        
        if y < 5:
            if len_genome - 1 and len_genome:
                z = randint(0,len_genome - 1)
                self.genome[z] = randint(max(self.max_neurons - 2,0),
                                         self.max_neurons + 2)
                
            elif len_genome :
                self.genome[0] = randint(max(self.max_neurons - 2,0),
                                         self.max_neurons + 2)
                
            else :
                self.genome.append(randint(1,self.max_neurons))
                
    def update_links(self,net):
        prd = 0
        links = 0
        layer = [2] + self.genome + [net.n_outputs_]
        for i in range(len(layer)-1):
            prd = layer[i]*layer[i+1]
            links = links + prd 
        self.links = links     
            
    def Print_RNet(self):
        print("Hidden Layer:",self.genome,sep = "\t")
        print("Fitness:",self.fitness,sep = "\t")
        
        
        
        
        
        
        
        
        
        
        