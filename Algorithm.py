import MLPClassifier_ as MLP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles 
from random import randint,uniform
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap
#from sklearn import datasets
    
def algorithm():
    
    #data for graphs
    lst_mean_fitness = []
    lst_best_fitness = []
    len_max = 20
    lst_len_score = [ [] for i in range(len_max)]
    lst_gen = []
    
    #Creating first generation
    generation = MLP.RNet_population(10)
    gen = 1
           
    #datasets selection
    dataset = make_moons(n_samples = 100, noise = 0.2,random_state=1)
    #dataset = make_circles(n_samples = 100,noise=0.2,factor = 0.4, random_state = 1)
    #rnstate = randint(1,100)
    
    #datasets make_moons or make_circles 
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=0)
    
    generation.pop_fitness(X_train, X_test, y_train, y_test)
    generation.shortBubbleSort()
    
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
    
    
           #per dataset digits:
#        digits = datasets.load_digits()   
#        n_samples = len(digits.images)
#        Data = digits.images.reshape((n_samples,-1))
#        Target = digits.target
#        Data_train, data_test, target_train, target_test = \
#            train_test_split(Data,Target,test_size =.4,random_state = 42)
    
    
    #Covergence conditions 
    while (generation.Best_Score() < 1 and 
           last_best[1] < 5):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=gen)
        
    ######################################################## 
        
        #Create new generation
        
        Next_gen = MLP.RNet_population(0)
        


        #Algoritmo 1         
#        Next_gen.RNpopulation.append(generation.RNpopulation[0])
#        Next_gen.RNpopulation.append(generation.RNpopulation[1])
#        
#        for i in range(2):
#            for j in range(i,i+2,1):
#                tpl = (generation.RNpopulation[i].
#                       Crossover1(generation.RNpopulation[j]))
#                Next_gen.RNpopulation.append(tpl[0])
#                Next_gen.RNpopulation.append(tpl[1])
                
        #####################################################
        
        #Algoritmo 2
        sons = MLP.RNet_population(0)
        for i in range(len(generation.RNpopulation)):
            for j in range(len(generation.RNpopulation)):
                
                tpl = (generation.RNpopulation[i].
                       Crossover1(generation.RNpopulation[j]))
                sons.RNpopulation.append(tpl[0])
                sons.RNpopulation.append(tpl[1])
                
        sons.pop_fitness(X_train, X_test, y_train, y_test)        
        sons.shortBubbleSort()
        
        for i in range(10):
            Next_gen.RNpopulation.append(sons[i])
        ###################################################
        
        generation = Next_gen
        del Next_gen   
        
        
        #Mutazioni casuali 
        for i in range(1,len(generation.RNpopulation)):
            generation.RNpopulation[i].Mutation()
            
    ####################################################################
        gen = gen + 1
        
        #Update fitness and sort
        generation.pop_fitness(X_train, X_test, y_train, y_test)
        generation.shortBubbleSort()
        
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
        
        #various printing 
        print()
        print ("Generazione",'\t', gen,sep = None)
        print("num_Individuals = ", len(generation.RNpopulation),'\n')
        #generation.Print_pop()
        print("************Best Net************")
        generation.Print_Best()
        print("********************************")
    

    #adjoustin lst_len_score:
    for i in range(len_max):
        for j in range(gen):
            lst_len_score[i][j] = lst_len_score[i][j] / 10
    
    #Graphs    
    plt.figure(figsize = (8,5))
    
    plt.subplot(211)
    plt.plot(lst_gen,lst_mean_fitness)
    plt.ylabel("mean fitness")
    
    
    plt.subplot(212)
    plt.plot(lst_gen,lst_best_fitness) 
    plt.ylabel("best fitness")
    plt.xlabel("generations")
    
    plt.figure(figsize =(8,5))
    
    plt.subplot(211)
    
    colors = ["blue","red","green","brown","pink","yellow","purple",
              "grey","black","orange","steelblue","navy","silver","darksalmon",
              "olive","plum","fuchsia","deeppink","springgreen","azure","tomato"]
    
    for i in range(0,len_max):
        labels = "lenght = " + str(i + 1)
        plt.plot(lst_gen,lst_len_score[i],color = colors[i] ,label = labels)
    plt.ylabel("hidden layer fractions")
    plt.xlabel("generations")
    plt.ylim(ymax = 1.0) 
    plt.legend(loc ="best",bbox_to_anchor=(1,1),
               fontsize = "xx-small",ncol = 1)
    
    #plt.show()
    ##########################################################################
        #Showing Classification
        
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
            
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)            
    
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
    ##########################################################################  
    
algorithm()
    
    



