from random import randint

result = 800

class Individual:
    
    NMAX = 10
    
    def __init__(self):
        self.fitness = 0
        self.numbers = []
        for i in range(self.NMAX):
            self.numbers.append(randint(0,100))
    
    def calfitness(self):
        res = 0
        for i in self.numbers:
            res = res + i
        self.fitness = abs(result - res)  #fitness al contrario
        
    def crossover(self,individual):
        cut = randint(1,self.NMAX - 1)
        son = Individual()
        son.numbers = self.numbers[:cut] + individual.numbers[cut:]
        return son
    
    def mutation(self):
        a = randint(0,self.NMAX - 1)
        self.numbers[a] = randint(0,100)
        
    def Print(self):
        print ("Individuo : ",self.numbers, sep= '\t')
        print ("Fitness : ",self.fitness, sep = '\t')
    
#    def getNumbers(self):
#        return self.numbers
          
#    def getFitness(self):
#        return self.fitness
    
    
    
class Population:
    
    def __init__(self,pop_number = 100):
        self.pop_number = pop_number
        self.population = []
        for i in range(pop_number):
           self.population.append(Individual())
           
    def pop_fitness(self):
        for ind in self.population:
            ind.calfitness()

    #pesante
    def order(self):  #ordino in modo crescente in base al fitness minore(best)
        self.pop_fitness()
        x = Individual()
        for i in range(self.pop_number):
            for j in range(i,self.pop_number,1):
                if self.population[i].fitness > self.population[j].fitness:
                    x = self.population[j]
                    self.population[j] = self.population[i]
                    self.population[i] = x
                    
    def BestFit(self):
        self.order()
        return self.population[0].fitness
        
    def NextGen(self):
        Next = []
        i = 0
        #Creo la nuova generazione
        while (i < self.pop_number/10 and 
               len(Next) <= self.pop_number):
            for j in range(i,int(self.pop_number/10),1):
                Next.append(self.population[i].crossover(self.population[j]))
        self.population = Next
        Next = []
        for ind in self.population:
            prb = randint(0,100)
            if (prb < 10):
                ind.mutation()
        
    def PrintBest(self):
        self.order()
        self.population[0].Print()
        
    def Print_pop(self):
        for ind in self.population:
            ind.Print()
            print()
            
    def fitness_medio(self):
        media = 0
        for i in self.population:
            i.calfitness()
            media = media + i.fitness
        return media/self.pop_number
        
        
        
    
            
        
        
        
        