import Network_classes as MLP
import functions as fnc
from sklearn.model_selection import train_test_split
 
def algorithm():
    
    reverse_state = False #usefull sorting option:
                          # True = increasing fitness (log_loss)
                          # False = decreasing fitness  (accuracy test) 
    
    #lists for graphs
    lst_mean_fitness = []
    lst_best_fitness = []
    len_max = 20
    lst_len_score = [ [] for i in range(len_max)]
    lst_gen = []
    
    #Creating first generation
    generation = MLP.RNet_population(5)
    gen = 1

    # name = make_moons, make_circles, make_circles+ and handwritten_digits(no images)
    data = fnc.select_dataset(name = "make_circles+", noise = 0.1, n_sample = 100)
    
    X, y , X_train, X_test, y_train, y_test, dataset = data
                  
    generation.pop_fitness( X_train, X_test, y_train, y_test)
    generation.RNpopulation = sorted(generation.RNpopulation, 
                                     key = lambda x: x.fitness, 
                                     reverse = reverse_state)
#    generation.shortBubbleSort()
    generation.update_mean_fitness()
    
    #create last_best for convergence condition 
    last_best = [generation.Best_Score(),0,generation.RNpopulation[0].genome]
    
    #update lists for graphs       
    lst_mean_fitness.append(generation.mean_fitness)
    lst_best_fitness.append(generation.Best_Score())
    lst_gen.append(gen)
    for i in range(len_max):
        count = 0
        for j in range(len(generation.RNpopulation)):
            if i + 1 == len(generation[j].genome):
                count += 1
        lst_len_score[i].append(count) 
     
    #print result and show classification
    fnc.printing(generation,gen)    
    fnc.classifier(generation = generation,dataset = dataset,
                   X = X,y = y,X_train =X_train, 
                   X_test = X_test,y_train = y_train,y_test = y_test)
    
    #Covergence condition
    while  last_best[1] < 5: 
        
        #change to the random_state for train and test set 
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2, random_state=gen)   
            
        #Create new generation.
        generation = fnc.selection_rnd(generation,X_train, X_test, y_train, y_test)       
#        generation = fnc.all_sons(generation,X_train, X_test, y_train, y_test,
#                                  reverse_state)
        
        #sort new generation and updates
        generation.RNpopulation = sorted(generation.RNpopulation, 
                                     key = lambda x: x.fitness, 
                                     reverse = reverse_state )
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
                if i + 1 == len(generation[j].genome):
                    count += 1
            lst_len_score[i].append(count)
        
        #various print
        fnc.printing(generation,gen)
        
        #showing classification
        fnc.classifier(generation = generation,dataset = dataset,
                   X = X,y = y,X_train =X_train, 
                   X_test = X_test,y_train = y_train,y_test = y_test)
    
    #normalize lst_len_score:
    lst_len_score = [[i/len(generation.RNpopulation) for i in lst] 
                                                for lst in lst_len_score]

    #Graphs    
    fnc.final_plot(lst_mean_fitness, lst_best_fitness, lst_len_score,lst_gen,len_max)
     
if __name__ == "__main__":
    algorithm()
    




