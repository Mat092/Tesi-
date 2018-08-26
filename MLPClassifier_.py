from sklearn.neural_network import MLPClassifier
from random import randint


class RNet_population:
    
    def __init__(self,num_individuals = 10):
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
    
    def medium_pop_fitness(self): #calcola il fitness medio della popolazione
        fitness_tot = 0
        for RandNet in self.RNpopulation:
            fitness_tot = fitness_tot + RandNet.fitness
        self.mean_fitness = round(fitness_tot / len(self.RNpopulation),2)
    
    def calc_proba(self):
        fitness_sum = 0.0
        for i in range(len(self.RNpopulation)):
            fitness_sum = fitness_sum + self.RNpopulation[i].fitness
        for i in range(len(self.RNpopulation)):
            self.RNpopulation[i].prob = round(self.RNpopulation[i].fitness / fitness_sum,
                                                2)
       
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
                    
    def shortBubbleSort(self): #ordina la popolazione per fitness
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
            
class Random_Network:
    
    def __init__(self):
        self.fitness = 0
        self.prob = 0
        self.genome = []
        self.Neurons_product = 0
        self.num_hidden_layers = randint(1,10)
        for n in range(self.num_hidden_layers):
            self.genome.append(randint(1,100))
            
    def RNfitness(self, X_train, X_test, y_train, y_test): 
        #Metodi di MLPCassifier 
        tpl = tuple(self.genome)
        net = MLPClassifier(hidden_layer_sizes = tpl, 
                            activation = 'relu',solver = 'adam',
                            alpha = 1, #???
                            learning_rate = 'constant' , learning_rate_init = 0.001,
                            random_state = 1,
                            early_stopping = True, validation_fraction = 0.1)
        net.fit(X_train, y_train)
        #scores = cross_val_score(net,X_train,y_train,cv = None)
        self.fitness = net.score(X_test,y_test)            
        
    def Crossover1(self, RandNet): #sommo i contributi tagliati in x e y 
        son1 = Random_Network()                  #dalle due network
        son2 = Random_Network()
        x = randint(0,len(self.genome))
        y = randint(0,len(RandNet.genome))
        son1.genome = self.genome[:x] + RandNet.genome[y:]
        son2.genome = RandNet.genome[:y] + self.genome[x:]  
        return son1,son2
    
    def Mutation(self):
        #mutazione, può anche allungare o accorciare la rete di 1 layer
        x = randint(0,100)
        y = randint(0,100)
        if x < 30:
            if x < 15:
                self.genome.append(randint(1,100))
            elif len(self.genome) - 1 and len(self.genome):
                z = randint(0,len(self.genome) - 1)
                del self.genome[z]
        if y < 30:
            if len(self.genome) - 1 and len(self.genome):
                z = randint(0,len(self.genome) - 1)
                self.genome[z] = randint(1,100)
            elif len(self.genome) :
                self.genome[0] = randint(1,100)
            else :
                self.genome.append(randint(1,100))
                
    def cal_Neurons_product(self):
        num = 1
        for ind in self.genome:
                num = num*ind
        self.Neurons_product = num 
                
    def Print_RNet(self):
        print("Hidden_Layer:",self.genome,sep = "\t")
        print("Fitness:",self.fitness,sep = "\t")
        
        
        
        
        
        
        
        
        
        
        