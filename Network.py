import numpy as np
import math

class Network():
    
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes[:-1] , sizes[1:])]
    
    def feedforward(self,a):
        for b , w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a  
    
def sigmoid(z):
    return 1/(1 +pow(math.e,-z))

class Perceptron():
    
    def __init__(self):
        self.bias = np.random.rand(1)
        self.weights = np.random.rand(2)
        
        