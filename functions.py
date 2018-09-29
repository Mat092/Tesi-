
import Network_classes as MLP
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles , load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def selection_rnd(generation,X_train, X_test, y_train, y_test):
    #Algoritmo 1, mi tengo 20% di elite, poi genero figli fino a riempire la pop.
    
    pop_len = len(generation.RNpopulation)
    Next_gen = MLP.RNet_population(0)
    
    for i in range(int(pop_len / 5)):       
        Next_gen.RNpopulation.append(generation.RNpopulation[i])
    
    i = 0
    while len(Next_gen.RNpopulation) < pop_len:
        tpl = (generation.RNpopulation[i].
               Crossover1(generation.RNpopulation[randint(0,pop_len - 1)]))
        tpl[0].Mutation()
        tpl[1].Mutation()
        tpl[0].RNfitness(X_train, X_test, y_train, y_test)
        tpl[1].RNfitness(X_train, X_test, y_train, y_test)
        Next_gen.RNpopulation.append(tpl[0])
        Next_gen.RNpopulation.append(tpl[1])
        i += 1
 
    return Next_gen

def all_sons(generation,X_train, X_test, y_train, y_test, reverse_state):
    
    #   Algoritmo 2,mantengo il 20% d'elite, genero ogni figlio possibile 
    #   e tengo gli N migliori per la next gen 
    pop_len = len(generation.RNpopulation)
    Next_gen = MLP.RNet_population(0)
    
    for i in range(int(pop_len / 5)):
        Next_gen.RNpopulation.append(generation.RNpopulation[i])
    
    sons = MLP.RNet_population(0)
    for i in range(pop_len):
        for j in range(pop_len):
            
            tpl = (generation.RNpopulation[i].
                   Crossover1(generation.RNpopulation[j]))
            tpl[0].Mutation()
            tpl[1].Mutation()
            sons.RNpopulation.append(tpl[0])
            sons.RNpopulation.append(tpl[1])
                    
    sons.pop_fitness(X_train, X_test, y_train, y_test)        
    #sons.shortBubbleSort()
    sons.RNpopulation = sorted(sons.RNpopulation, 
                                 key = lambda x: x.fitness, 
                                 reverse = reverse_state )
    
    while len(Next_gen.RNpopulation) < pop_len:
        Next_gen.RNpopulation.append(sons[i])
        
    return Next_gen
 
def printing(generation,gen):
    print()
    print ("Generazione",'\t', gen,sep = None)
    print("num_Individuals = ", len(generation.RNpopulation))
    #generation.Print_pop()
    print("************Best Net************")
    generation.Print_Best()
    print("********************************")
    
def select_dataset(name, noise = 0, n_sample = 100):
    
    if name == "moons":
        dataset = make_moons(n_samples = n_sample, noise = noise,random_state=1)
        X, y = dataset 
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2, random_state=0)
        return X, y , X_train, X_test, y_train, y_test, dataset
    
    elif name == "circles":
        dataset = make_circles(n_samples = n_sample,
                               noise=noise,
                               factor = 0.6,
                               random_state = 1)
        X, y = dataset 
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2, random_state=0)
        return X, y , X_train, X_test, y_train, y_test, dataset

    elif name == "circles+":
        dataset1 = make_circles(n_samples = n_sample,noise=noise,
                                factor = 0.6, random_state = 1)
        dataset2 = make_circles(n_samples = int(n_sample/2), noise=noise,
                                factor = 0.3, random_state = 1)
        for i in range(len(dataset2[1])):
            dataset2[1][i] = 0
        X1,y1 = dataset1
        X2,y2 = dataset2
        X = np.concatenate((X1,X2),axis = 0)
        y = np.concatenate((y1,y2),axis = 0)
        dataset = X, y
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.2, random_state=0)
        return X, y , X_train, X_test, y_train, y_test, dataset
    
    elif name == "handwritten_digits":
            digits = load_digits()
            n_samples = len(digits.images)
            X = digits.images.reshape((n_samples,-1))
            y = digits.target
            dataset = (X,y)
            X_train, X_test, y_train, y_test = \
                train_test_split(X,y,test_size =.2,random_state = 1)
            return X, y , X_train, X_test, y_train, y_test, dataset
    else:
        print("Error: no dataset called : ", name )
        exit()

def final_plot(lst_mean_fitness, lst_best_fitness, lst_mean_links, 
               lst_len_score,lst_gen,len_max):
    
    plt.figure(figsize = (8,5))
    
    plt.subplot(111)
    plt.plot(lst_gen,lst_mean_fitness,marker = 'o')
    plt.ylabel("mean fitness")
    
    plt.figure(figsize = (8,5))
    
    plt.subplot(111)
    plt.plot(lst_gen,lst_best_fitness,marker = 'o') 
    plt.ylabel("best fitness")
    plt.xlabel("generations")
    
    plt.figure(figsize = (8,5))
    
    plt.subplot(111)
    plt.plot(lst_gen,lst_mean_links,marker = 'o')
    plt.ylabel("mean links")
    plt.xlabel("generations")
    
    #plt.figure(figsize =(7,5))
    
    
    
#    
#    plt.subplot(211)
#    
#    colors = ["blue","red","green","brown","pink","yellow","purple",
#              "grey","black","orange","steelblue","navy","silver","darksalmon",
#              "olive","plum","fuchsia","deeppink","springgreen","azure","tomato"]
#    
#    for i in range(0,len_max):
#        labels = "lenght = " + str(i + 1)
#        plt.plot(lst_gen,lst_len_score[i],
#                 color = colors[i],
#                 label = labels)
#    plt.ylabel("hidden layer fractions")
#    plt.xlabel("generations")
#    #plt.ylim(ymax = 1.0) 
#    plt.legend(loc ="best",bbox_to_anchor=(1,1),
#               fontsize = "x-small",ncol = 1)
    
    plt.tight_layout()
    plt.show()  
      
def classifier(generation,dataset,X,y,X_train,y_train,X_test,y_test,gen):
    figure = plt.figure(figsize = (8,5))   

    classifiers = [generation.RNpopulation[0]]
    
    datasets = [dataset]  
    names = ["generazione " +  str(gen)]
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
        #decomment to generate input data
#        if ds_cnt == 0:
#            ax.set_title("Input data")
#        
#        # Plot the training points
#        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
#                   edgecolors='k')
#       
#        # and testing points
#        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
#                   edgecolors='k')
#        ax.set_xlim(xx.min(), xx.max())
#        ax.set_ylim(yy.min(), yy.max())
#        ax.set_xticks(())
#        ax.set_yticks(())
#        i += 1
    
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            
            #clf.fit(X_train, y_train)
            score = clf.fitness            
    
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
          
            #print("da qui ci passo")
            Z = clf.scorer.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
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
            
#            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
#                    size=15, horizontalalignment='right')
            i += 1
        
    plt.tight_layout()
    plt.show()
    
def update_small_layer(lst,generation):
    if len(generation[0].genome):
        small_layer = min(generation.RNpopulation[0].genome)
        lst.append(small_layer)
    else :
        lst.appen(0)
    
    
    