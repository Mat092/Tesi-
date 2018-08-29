from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles #, make_classification
#from sklearn import datasets
from random import randint,random
import MLPClassifier_ as MLP
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def algorithm():
    lst_mean_fitness = []
    lst_best_fitness = []
    lst_gen = []
    generation = MLP.RNet_population(10)
    gen = 1
    
    #dataset = make_circles(n_samples = 100,noise=0.2,factor = 0.4, random_state = 1)
    #rnstate = randint(1,100)
    dataset = make_moons(n_samples = 100, noise = 0.2,random_state=1)
        
        
    #per datasets make_moons e make_circles 
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=1)
        
    #        #per dataset digits:
#        digits = datasets.load_digits()   
#        n_samples = len(digits.images)
#        Data = digits.images.reshape((n_samples,-1))
#        Target = digits.target
#        Data_train, data_test, target_train, target_test = \
#            train_test_split(Data,Target,test_size =.4,random_state = 42)
    
    generation.pop_fitness(X_train, X_test, y_train, y_test)
    generation.medium_pop_fitness()
    generation.shortBubbleSort()
   
    lst_mean_fitness.append(generation.mean_fitness)
    lst_best_fitness.append(generation.Best_Score())
    lst_gen.append(gen)
   
    print ('\n',"Generazione",'\t', gen,sep = None)
    print("num_Individuals = ", generation.num_individuals)
    #generation.Print_pop()
    print("************Best Net**********")
    generation.Print_Best()
    print("*******************************") 
    
    
    
    while generation.Best_Score() < 1: #continua fino a fitness = 1
        
        generation.calc_proba()
        Next_gen = MLP.RNet_population(0)
        
#######################################################################################        
         #Scelta casuale dei genitori basata sul fitness, ma meglio ! Roulette wheel
        
        while len(Next_gen.RNpopulation) < len(generation.RNpopulation):
           
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
                        
                        break
                    elif j == exclude:
                        rnd = round(random(),2)
                    previous_proba = generation.RNpopulation[i].prob
                    
            # pezza nel caso non prendesse 2 genitori, ci passa al massimo 1 o 2volte
            # per generazione quindi non mi preoccupa 
            while len(parents) < 2:
                    rnd = randint(0,9)
                    parents.append(generation.RNpopulation[rnd])
                   
                    
                  
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
        
        generation.pop_fitness(X_train, X_test, y_train, y_test)
        
        #Calcolo fitness e ordino gli individui 
        generation.medium_pop_fitness()
        generation.shortBubbleSort()
        gen = gen + 1
        
        lst_mean_fitness.append(generation.mean_fitness)
        lst_best_fitness.append(generation.Best_Score())
        lst_gen.append(gen)
        
        
        print ("Generazione",'\t', gen,sep = None)
        print("num_Individuals = ", len(generation.RNpopulation),'\n')
        #generation.Print_pop()
        print("************Best Net**********")
        generation.Print_Best()
        print("********************************")
        
    plt.figure()
    plt.subplot(211)
    plt.plot(lst_gen,lst_mean_fitness)
    
    plt.subplot(212)
    plt.plot(lst_gen,lst_best_fitness) 
    plt.show()
##############################################################################
        #GRAFICI
    figure = plt.figure()   
    tpl = tuple(generation.RNpopulation[0].genome)
    
    classifiers = [MLPClassifier(hidden_layer_sizes = tpl, 
                            activation = 'relu',solver = 'adam',
                            alpha = 1, #???
                            learning_rate = 'constant' , learning_rate_init = 0.001,
                            random_state = 1,
                            early_stopping = False, validation_fraction = 0.1)]
    datasets = [dataset]  
    names = ["Neural Net"]
    h = .02
    i = 1
    
    for ds_cnt, ds in enumerate(datasets):
    

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
    
        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
                                    
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
       
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
    
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            
            ###############################################################
            #è praticamente il mio fitness!
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test) #molto importante!!!
    
            ################################################################
    
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                #print("da qui ci passo")
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)
    
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            #scrive lo "score"
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
        
    plt.tight_layout()
    plt.show()
###############################################################################  
    
algorithm()


    



