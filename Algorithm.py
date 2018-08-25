from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles#, make_classification
from sklearn import datasets
from random import randint,random
import MLPClassifier_ as MLP

def algorithm():
    generation = MLP.RNet_population(10)
    gen = 1
    
    #dataset = make_circles(n_samples = 100,noise=0.2,factor = 0.5, random_state = 1)
    #rnstate = randint(1,100)
    dataset = make_moons(n_samples = 100, noise = 0.2,random_state=1)
        
        
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
           generation.pop_fitness(X_train, X_test, y_train, y_test,X,y)
           generation.medium_pop_fitness()
           generation.shortBubbleSort()
           print ('\n',"Generazione",'\t', gen,sep = None)
           print("num_Individuals = ", generation.num_individuals)
           #generation.Print_pop()
           print("************Best Net**********")
           generation.Print_Best()
           print("*******************************") 
           if generation.Best_Score() == 1:
               break
        
        generation.calc_proba()
        Next_gen = MLP.RNet_population(0)
        
#######################################################################################        
         #Scelta casuale dei genitori basata sul fitness, ma meglio ! Roulette wheel
        
        while len(Next_gen.RNpopulation) < len(generation.RNpopulation):
            temp_list = []
            exclude = -1
            parents = [] 
            for i in range(2):
                
                previous_proba = 1.0
                rnd = round(random(),2) #cas number 0 < x < 1
                
                for j in range(len(generation.RNpopulation)):
                    proba = generation.RNpopulation[j].prob
                    if (proba <= rnd and rnd <= previous_proba and 
                        j != exclude) :
                        parents.append(generation.RNpopulation[j])
                        exclude = j
                        temp_list.append(j)
                        break
                    elif j == exclude:
                        rnd = round(random(),2)
                    previous_proba = generation.RNpopulation[i].prob
                    
            # pezza nel caso non prendesse 2 genitori, ci passa al massimo 1 o 2volte
            # per generazione quindi non mi preoccupa 
            while len(parents) < 2:
                    rnd = randint(0,9)
                    parents.append(generation.RNpopulation[rnd])
                    temp_list.append(rnd)
                    
            print(temp_list)       
            tpl = parents[0].Crossover1(parents[1])
            Next_gen.RNpopulation.append(tpl[0])
            Next_gen.RNpopulation.append(tpl[1])
        
########################################################################################
#        #Scelta casuale dei genitori basata sul fitness
#        while len(Next_gen.RNpopulation) < len(generation.RNpopulation):
#            x = randint(0,100) / 100
#            previous_proba = 1.0
#            for i in range(len(generation.RNpopulation)-1):
#                if generation.RNpopulation[i].prob < x < previous_proba :
#                    y = randint(0,100) / 100
#                    previous_proba2 = 1.0
#                    for j in range(len(generation.RNpopulation)-1):
#                        if (generation.RNpopulation[j].prob < y < previous_proba2 ) :
#                            tpl = generation.RNpopulation[i].Crossover1(generation.RNpopulation[j])
#                            Next_gen.RNpopulation.append(tpl[0])
#                            Next_gen.RNpopulation.append(tpl[1])
#                        previous_proba2 = generation.RNpopulation[j].prob
#                previous_proba = generation.RNpopulation[i].prob
                
###################################################################################################            
                    
                #Selezione randomica del secondo elemento
                #il primo va in ordine 
    
#        for i in range(len(generation.RNpopulation)):
#            if len(Next_gen.RNpopulation) >= len(generation.RNpopulation):
#                break
#            rnd = randint(i,len(generation.RNpopulation)-1)
#            tpl = generation.RNpopulation[i].Crossover1(generation.RNpopulation[rnd])
#            Next_gen.RNpopulation.append(tpl[0])
#            Next_gen.RNpopulation.append(tpl[1])
                
######################################################################################
                
        generation = Next_gen
        del Next_gen
        
        #Mutazioni casuali 
        for i in range(len(generation.RNpopulation)):
            generation.RNpopulation[i].Mutation()
        
        generation.pop_fitness(X_train, X_test, y_train, y_test,X,y)
        
        #Calcolo fitness e ordino gli individui 
        generation.medium_pop_fitness()
        generation.shortBubbleSort()
        
        gen = gen + 1
        print ("Generazione",'\t', gen,sep = None)
        print("num_Individuals = ", len(generation.RNpopulation),'\n')
        #generation.Print_pop()
        print("************Best Net**********")
        generation.Print_Best()
        print("********************************")
        

algorithm()


    



