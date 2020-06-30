import numpy as np

class Adam():
    
    """Adam Optimizer """
    
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-7
        self.v = dict()
        self.s = dict()
     
    def optimizer_name(self):
        return "Adam"
    
    def init_params(self,layer_dims):
        
        for i in range(1,len(layer_dims)):
            
            self.v["dW" + str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
            self.v["db" + str(i)] = np.zeros((layer_dims[i], 1))
            
            self.s["dW" + str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
            self.s["db" + str(i)] = np.zeros((layer_dims[i], 1))
            
    def update_params_using_Adam(self, grads, t, layer_dims,parameters):
        
        new_learning_rate = self.learning_rate * np.divide(np.sqrt(1 - np.power(self.beta2, t)), 1- np.power(self.beta1, t))
        
        new_parameters = dict()
        
        for i in range(1, len(layer_dims)):
        
            self.v["dW" + str(i)] = self.beta1 * self.v["dW" + str(i)] + (1 - self.beta1) * grads["dW" + str(i)]

            self.v["db" + str(i)] = self.beta1 * self.v["db" + str(i)] + (1 - self.beta1) * grads["db" + str(i)]

            self.s["dW" + str(i)] = self.beta2 * self.s["dW" + str(i)] + (1 - self.beta2) * grads["dW" + str(i)] ** 2

            self.s["db" + str(i)] = self.beta2 * self.s["db" + str(i)] + (1 - self.beta2) * grads["db" + str(i)] ** 2

            new_parameters["W" + str(i)] = parameters["W" + str(i)] - new_learning_rate * np.divide(self.v["dW" + str(i)], np.sqrt(self.s["dW" + str(i)]) + self.epsilon) 
            
            new_parameters["b" + str(i)] = parameters["b" + str(i)] - new_learning_rate * np.divide(self.v["db" + str(i)], np.sqrt(self.s["db" + str(i)]) + self.epsilon)
            
        return new_parameters
    

    
class Adamax():
    
    """Adamax Optimizer """

    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-7
        self.v = dict()
        self.s = dict()
     
    def optimizer_name(self):
        return "Adamax"
    
    def init_params(self,layer_dims):
        
        for i in range(1,len(layer_dims)):
            
            self.v["dW" + str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
            self.v["db" + str(i)] = np.zeros((layer_dims[i], 1))
            
            self.s["dW" + str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
            self.s["db" + str(i)] = np.zeros((layer_dims[i], 1))
            
    def update_params_using_Adamax(self, grads, t, layer_dims,parameters):
        
        new_learning_rate = self.learning_rate / (1 - self.beta1 ** t)
        
        new_parameters = dict()
                
        for i in range(1, len(layer_dims)):
        
            self.v["dW" + str(i)] = self.beta1 * self.v["dW" + str(i)] + (1 - self.beta1) * grads["dW" + str(i)]

            self.v["db" + str(i)] = self.beta1 * self.v["db" + str(i)] + (1 - self.beta1) * grads["db" + str(i)]

            self.s["dW" + str(i)] = np.maximum(self.beta2 * self.s["dW" + str(i)], np.absolute(grads["dW" + str(i)]))

            self.s["db" + str(i)] = np.maximum(self.beta2 * self.s["db" + str(i)], np.absolute(grads["db" + str(i)]))
           
            new_parameters["W" + str(i)] = parameters["W" + str(i)] - new_learning_rate * np.divide(self.v["dW" + str(i)], self.s["dW" + str(i)] + self.epsilon) 
            
            new_parameters["b" + str(i)] = parameters["b" + str(i)] - new_learning_rate * np.divide(self.v["db" + str(i)], self.s["db" + str(i)] + self.epsilon)
            
        return new_parameters
    
    
class RmsProp():
    
    """RmsProp Optimizer """
    
    def __init__(self,learning_rate = 0.1,beta = 0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = 1e-8
        self.v = dict()
        
    def optimizer_name(self):
        return "RMS"
        
    def init_params(self,layer_dims):

        for i in range(1,len(layer_dims)):

            self.v["dW" + str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
            self.v["db" + str(i)] = np.zeros((layer_dims[i], 1))

    def update_params_using_RmsProp(self,grads,layer_dims,parameters):
        new_parameters = dict()

        for i in range(1,len(layer_dims)):
            
            self.v["dW" + str(i)] = self.beta * self.v["dW" + str(i)] + (1 - self.beta) * grads["dW" + str(i)]

            self.v["db" + str(i)] = self.beta * self.v["db" + str(i)] + (1 - self.beta) * grads["db" + str(i)]

            new_parameters["W" + str(i)] = parameters["W" + str(i)] - self.learning_rate * self.v["dW" + str(i)]

            new_parameters["b" + str(i)] = parameters["b" + str(i)] - self.learning_rate * self.v["db" + str(i)]
            
        return new_parameters