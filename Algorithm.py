from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles#, make_classification
from sklearn import datasets
from random import randint
import MLPClassifier_ as MLP

def algorithm():
    generation = MLP.RNet_population(4)
    gen = 1
    
    #dataset = make_circles(n_samples = 100,noise=0.2,factor = 0.5, random_state = 1)
    #rnstate = randint(1,100)
    dataset = make_moons(n_samples = 100, noise = 0.27,random_state=1)
        
        
    #per datasets
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=1)
        
    #        #per numeri scritti a mano:
#        digits = datasets.load_digits()   
#        n_samples = len(digits.images)
#        Data = digits.images.reshape((n_samples,-1))
#        Target = digits.target
#        Data_train, data_test, target_train, target_test = \
#            train_test_split(Data,Target,test_size =.4,random_state = 42)
    
    while generation.Best_Score() < 1: #continua fino a fitness = 1
        
        if gen == 1: 
           generation.pop_fitness(X_train, X_test, y_train, y_test)
           generation.medium_pop_fitness()
           generation.shortBubbleSort()
           print ('\n',"Generazione",'\t', gen)
           print("num_Individuals = ", generation.num_individuals,'\n')
           #generation.Print_pop()
           print("************Best Net**********")
           generation.Print_Best()
           print("********************************")
        
        Next_gen = MLP.RNet_population(0)
        #Creazione della nuova generazione
        
        for i in range(len(generation.RNpopulation)):
            if len(Next_gen.RNpopulation) >= len(generation.RNpopulation):
                break
            rnd = randint(i,len(generation.RNpopulation)-1)
            tpl = generation.RNpopulation[i].Crossover1(generation.RNpopulation[rnd])
            Next_gen.RNpopulation.append(tpl[0])
            Next_gen.RNpopulation.append(tpl[1])
            
            
#            for j in range(i + 1,len(generation.RNpopulation)):
#                if len(Next_gen.RNpopulation) >= len(generation.RNpopulation):
#                    break
#                tpl =generation.RNpopulation[i].Crossover1(generation.RNpopulation[j])
#                Next_gen.RNpopulation.append(tpl[0])
#                Next_gen.RNpopulation.append(tpl[1])
                
        
#        #Creazione della nuova generazione
#        parents = MLP.RNet_population(0)
#        for i in range(generation.num_individuals):    #Seleziono i genitori
#            prb = randint(0,85) / 100
#            if generation[i].fitness > prb:
#                parents.RNpopulation.append(generation[i])
#        parents.num_individuals = len(parents.RNpopulation)   
#        
#        for i in range(parents.num_individuals):
#            if len(Next_gen.RNpopulation) >= generation.num_individuals:
#                break
#            for j in range(i,parents.num_individuals):
#                if len(Next_gen.RNpopulation) >= generation.num_individuals:
#                    break
#                tpl = parents[i].Crossover1(parents[j])
#                Next_gen.RNpopulation.append(tpl[0])
#                Next_gen.RNpopulation.append(tpl[1])
        
#        Next_gen.num_individuals = len(Next_gen.RNpopulation) 
            
        generation = Next_gen
        del Next_gen
        
        #Mutazioni casuali 
        for i in range(generation.num_individuals):
            generation.RNpopulation[i].Mutation()
        
        generation.pop_fitness(X_train, X_test, y_train, y_test)
        
        #Calcolo fitness e ordino gli individui 
        generation.medium_pop_fitness()
        generation.shortBubbleSort()
        
        gen = gen + 1
        print ("Generazione",'\t', gen)
        print("num_Individuals = ", len(generation.RNpopulation),'\n')
        #generation.Print_pop()
        print("************Best Net**********")
        generation.Print_Best()
        print("********************************")
        

algorithm()


    



