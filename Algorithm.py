from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles#, make_classification
from sklearn import datasets
#from random import randint
import MLPClassifier_ as MLP

def algorithm():
    generation = MLP.RNet_population()
    gen = 1
    
    while generation.Best_Score() < 1: #continua fino a fitness = 1
        
        
        dataset = make_moons(n_samples = 1000, noise=0.3, random_state=0)
        digits = datasets.load_digits()
        #make_circles(noise=0.2, factor=0.5, random_state=1)
        
        #per make_moons
#        X, y = dataset
#        X = StandardScaler().fit_transform(X)
#        X_train, X_test, y_train, y_test = \
#            train_test_split(X, y, test_size=.4, random_state=42)
        
        #per numeri scritti a mano:
        n_samples = len(digits.images)
        Data = digits.images.reshape((n_samples,-1))
        Target = digits.target
        Data_train, data_test, target_train, target_test = \
            train_test_split(Data,Target,test_size =.4,random_state = 42)
    
        if gen == 1: 
           #generation.Print_pop() 
           generation.pop_fitness(Data_train, data_test, target_train, target_test)
           generation.shortBubbleSort()
        
        #Creazione della nuova generazione
        Next_gen = MLP.RNet_population()
        for i in range(generation.num_individuals):
            if len(Next_gen.RNpopulation) >= generation.num_individuals:
                break
            for j in range(i,generation.num_individuals):
                if len(Next_gen.RNpopulation) >= generation.num_individuals:
                    break
                tpl = generation.RNpopulation[i].Crossover1(generation.RNpopulation[j])
                Next_gen.RNpopulation.append(tpl[0])
                Next_gen.RNpopulation.append(tpl[1])
        
        generation = Next_gen
        del Next_gen
        
        #Mutazioni casuali 
        for i in range(generation.num_individuals):
            generation.RNpopulation[i].Mutation()
        
        generation.pop_fitness(Data_train, data_test, target_train, target_test)
        
        #Calcolo fitness e ordino gli individui 
        generation.medium_pop_fitness()
        generation.shortBubbleSort()
        print('\n',"************Best Net**********")
        print ("Generazione",'\t', gen)
        #generation.Print_pop()
        generation.Print_Best()
        print("********************************")
        gen = gen + 1

algorithm()

