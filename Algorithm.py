import Network_classes as MLP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles 
from random import randint,uniform
from sklearn.neural_network import MLPClassifier
from selections import selection1, classifier, all_sons

#from sklearn import datasets
  
def algorithm():
    
    #data for graphs
    lst_lenghts = []
    lst_mean_fitness = []
    lst_best_fitness = []
    lst_fitness = []
    len_max = 20
    lst_len_score = [ [] for i in range(len_max)]
    lst_gen = []
    
    #Creating first generation
    generation = MLP.RNet_population(20)
    gen = 1
           
    #datasets selection
    #dataset = make_moons(n_samples = 100, noise = 0.2,random_state=1)
    dataset1 = make_circles(n_samples = 100,noise=0.1,factor = 0.6, random_state = 1)
    dataset2 = make_circles(n_samples = 50,noise=0.05,factor = 0.3, random_state = 1)
    #rnstate = randint(1,100)
    
    #datasets make_moons 
#    X, y = dataset1
#    X = StandardScaler().fit_transform(X)
#    X_train, X_test, y_train, y_test = \
#            train_test_split(X, y, test_size=.4, random_state=0)
     
    #datasets per make_circles+
    for i in range(len(dataset2[1])):
        dataset2[1][i] = 0
    X1,y1 = dataset1
    X2,y2 = dataset2
    X = np.concatenate((X1,X2),axis = 0)
    y = np.concatenate((y1,y2),axis = 0)
    dataset = X, y
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=0)
            
             
    generation.pop_fitness(X_train, X_test, y_train, y_test)
    #generation.shortBubbleSort()
    generation.RNpopulation = sorted(generation.RNpopulation, 
                                     key = lambda x: x.fitness, reverse = True )
    generation.update_mean_fitness()
    
    last_best = [generation.Best_Score(),0,generation.RNpopulation[0].genome]
            
    lst_mean_fitness.append(generation.mean_fitness)
    lst_best_fitness.append(generation.Best_Score())
    lst_gen.append(gen)
    for i in range(len_max):
        count = 0
        for j in range(len(generation.RNpopulation)):
            if i + 1 == len(generation.RNpopulation[j].genome):
                count += 1
        lst_len_score[i].append(count) 
           
    print ('\n',"Generazione",'\t', gen,sep = None)
    print("num_Individuals = ", generation.num_individuals)
    #generation.Print_pop()
    print("************Best Net**********")
    generation.Print_Best()
    print("*******************************")
    #print(lst_lenghts)
    classifier(generation = generation,dataset = dataset,
                   X = X,y = y,X_train =X_train, 
                   X_test = X_test,y_train = y_train,y_test = y_test)
    
           #per dataset digits:
#        digits = datasets.load_digits()   
#        n_samples = len(digits.images)
#        Data = digits.images.reshape((n_samples,-1))
#        Target = digits.target
#        Data_train, data_test, target_train, target_test = \
#            train_test_split(Data,Target,test_size =.4,random_state = 42)
    
    #Covergence condition
    while  last_best[1] < 5: 
        
        #change to the random_state for train and test set 
#        X_train, X_test, y_train, y_test = \
#            train_test_split(X, y, test_size=.4, random_state=gen)   
            
        #Create new generation
        
        generation = selection1(generation,X_train, X_test, y_train, y_test)       
        #generation = all_sons(generation,X_train, X_test, y_train, y_test)
        
        #sort new generation and updates
        generation.RNpopulation = sorted(generation.RNpopulation, 
                                     key = lambda x: x.fitness, reverse = True )
        generation.num_individuals = len(generation.RNpopulation)
        generation.update_mean_fitness()
        
        #update gen 
        gen = gen + 1
        
        #Update last_best 
        if (last_best[0] == generation.Best_Score() or 
            last_best[2] == generation.RNpopulation[0].genome):
            last_best[1] += 1
        else:
            last_best[1] = 0
        last_best[0] = generation.Best_Score()
        last_best[2] = generation.RNpopulation[0].genome
        
        #update list for graphs 
        lst_mean_fitness.append(generation.mean_fitness)
        lst_best_fitness.append(generation.Best_Score())
        lst_gen.append(gen)
        for i in range(len_max):
            count = 0
            for j in range(len(generation.RNpopulation)):
                if i + 1 == len(generation.RNpopulation[j].genome):
                    count += 1
            lst_len_score[i].append(count)
        for i in range(len(generation.RNpopulation)):
            lst_lenghts.append(len(generation.RNpopulation[i].genome))
        
        #various printing 
        print()
        print ("Generazione",'\t', gen,sep = None)
        print("num_Individuals = ", len(generation.RNpopulation),'\n')
        #generation.Print_pop()
        print("************Best Net************")
        generation.Print_Best()
        print("********************************")
        
        #showing classification
        classifier(generation = generation,dataset = dataset,
                   X = X,y = y,X_train =X_train, 
                   X_test = X_test,y_train = y_train,y_test = y_test)
        
    #adjoustin lst_len_score:
    for i in range(len_max):
        for j in range(gen):
            lst_len_score[i][j] = lst_len_score[i][j] / len(generation.RNpopulation)
    
    #Graphs    
    plt.figure(figsize = (8,5))
    
    plt.subplot(211)
    plt.plot(lst_gen,lst_mean_fitness,marker = 'o')
    plt.ylabel("mean fitness")
    
    plt.subplot(212)
    plt.plot(lst_gen,lst_best_fitness,marker = 'o') 
    plt.ylabel("best fitness")
    plt.xlabel("generations")
    
    plt.figure(figsize =(8,10))
    
    plt.subplot(211)
    
    colors = ["blue","red","green","brown","pink","yellow","purple",
              "grey","black","orange","steelblue","navy","silver","darksalmon",
              "olive","plum","fuchsia","deeppink","springgreen","azure","tomato"]
    
    for i in range(0,len_max):
        labels = "lenght = " + str(i + 1)
        plt.plot(lst_gen,lst_len_score[i],
                 color = colors[i],
                 label = labels)
    plt.ylabel("hidden layer fractions")
    plt.xlabel("generations")
    plt.ylim(ymax = 1.0) 
    plt.legend(loc ="best",bbox_to_anchor=(1,1),
               fontsize = "xx-small",ncol = 1)
     
if __name__ == "__main__":
    algorithm()
    




