from random import randint

result = "SoNo Un4 sTr1nGa Qu4lUnQu3."

class Elemento:  
    
    lenght = len(result) #per ora mi limito a lunghezze fisse, date da "result"
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz1234567890,."
    
    def __init__(self):  #inizializzo fit e creo stringa casuale, di len giusta
        self.fit = 0
        self.string = ""
        for i in range(self.lenght):
            self.string = self.string  + self.L[randint(0,len(self.L)-1)]

    def crossover(self,El): #Funzionante
        res = Elemento()
        cut = randint(0,self.lenght)
        res.string = self.string[:cut] + El.string[cut:]
        return res      #nuovo elemento dal crossover di due 
    
    def mutation(self): #Funzionante
        x = randint(0,self.lenght - 1)
        y = randint(0,len(self.L) - 1)
        L1 = list(self.string)
        L2 = list(self.L)
        L1[x] = L2[y]
        self.string = "".join(L1)
        
    def calcfitness(self,result): #Funzionante
        self.fit = 0
        dF = 1.0/self.lenght
        for i in range(self.lenght):
            if self.string[i] == result[i]:
                self.fit += dF
                
    def order(self,NMAX, List): #Funzionante
        x = Elemento()
        i = 0
        while i < NMAX:
            j = i
            while j < NMAX:
                if List[i].fit < List[j].fit:
                    x = List[j]
                    List[j] = List[i]
                    List[i] = x
                j = j + 1
            i = i + 1
            
    def Print(self):  #Funzionante
        print("Fitness = ",self.fit)
        print("String = ", self.string,'\n')

        
        
        