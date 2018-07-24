import classElemento as elem 
from random import randint 


def best_fitness(List,NMAX):
    x = elem.Elemento()
    x.order(NMAX,List)
    List[0].Print()

def breed(NMAX):
    prova = elem.Elemento()
    generazione = []
    for j in range(NMAX):
        generazione.append(elem.Elemento())
    gen = 1
    while gen :
        #Calcolo fitness e ordinamento
        for j in range(NMAX):
            generazione[j].calcfitness(elem.result)
            if generazione[j].fit >= 0.999999:
                print("Ecco il risultato dalla generazione: ",gen + 1)
                generazione[j].Print()
                return 1
            
        prova.order(NMAX,generazione)
        
        #applico le modifiche 
        Next = []
        for j in range(10):
            for i in range(10):
                #crossover
                    x = elem.Elemento()
                    x = generazione[j].crossover(generazione[i])
                    Next.append(x)
        generazione = Next
        Next = []
        
        for figlio in generazione:    
            #mutazione
            prb = randint(0,10) / 10
            if (prb < 0.3):
                figlio.mutation()
        gen += 1
              
breed(100)
        
                
            
            
            
            
            
            
            
                
        
                
                
                
    
        
    
    
        
    
    