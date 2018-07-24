import classes as cl
import matplotlib as mp
import numpy as np

def Algorithm():
    gen = 1
    fitness_medi = np.array([])
    GenList = np.array([]) 
    generazione = cl.Population(10)
    
    while generazione.BestFit():
        generazione.NextGen()
        
        fitness_medi = np.append(fitness_medi, generazione.fitness_medio())
        GenList = np.append(GenList, gen)
        
        gen +=1
        
    print ("Ho trovato il risultato alla generazione: ", gen)
    generazione.population[0].Print()
    
    mp.pyplot.plot(GenList, fitness_medi)
    
Algorithm()   
    

def Analisi(N_pop):
    pops_fitness = []
    Generazioni = []
    for i in range(N_pop):
        Generazioni.append(cl.Population())
        pops_fitness.append(Generazioni[i].fitness_medio())
    


        
    




