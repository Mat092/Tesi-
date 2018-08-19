from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons#, make_circles, make_classification
#from random import randint
import MLPClassifier_ as MLP


def algorithm():
    generation = MLP.RNet_population()
    gen = 1
    while generation.Best_Score() < 1:
        
        dataset = make_moons(noise=0.3, random_state=0)
        X, y = dataset
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
    
        if gen == 1: 
           #generation.Print_pop() 
           generation.pop_fitness(X_train, X_test, y_train, y_test)
           generation.shortBubbleSort()
        
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
        
        for i in range(generation.num_individuals):
            generation.RNpopulation[i].Mutation()
        
        generation.pop_fitness(X_train, X_test, y_train, y_test)
        
        generation.medium_pop_fitness()
        generation.shortBubbleSort()
        print('\n',"************Best Net**********")
        print ("Generazione",'\t', gen)
        #generation.Print_pop()
        generation.Print_Best()
        print("********************************")
        gen = gen + 1
      
algorithm()        
