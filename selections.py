from matplotlib.colors import ListedColormap
import Network_classes as MLP
import matplotlib.pyplot as plt
import numpy as np
from random import randint, uniform 

def all_sons(generation,X_train, X_test, y_train, y_test):
    
    #   Algoritmo 2, genero ogni figlio possibile 
    #   e tengo gli N migliori per la next gen 
    pop_len = len(generation.RNpopulation)
    Next_gen = MLP.RNet_population(0)
    for i in range(int(pop_len / 10)):
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
                                 key = lambda x: x.fitness, reverse = True )
    
    for i in range(pop_len - 5):
        Next_gen.RNpopulation.append(sons[i])
        
    return Next_gen


def selection1(generation,X_train, X_test, y_train, y_test):
    #Algoritmo 1, mi tengo 20% di elite, poi genero figli
    
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
    
    
def classifier(generation,dataset,X,y,X_train,y_train,X_test,y_test):
    figure = plt.figure(figsize = (8,5))   

    classifiers = [generation.RNpopulation[0]]
    
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
            #scrive lo "score"
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
        
    plt.tight_layout()
    plt.show()