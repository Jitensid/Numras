

import numpy as np
import math
from sklearn.metrics import accuracy_score

class Numras():
    
    def __init__(self, layer_dims, activation_functions, parameter_type = "default",optimizer = None):
        self.layer_dims = layer_dims
        self.L = len(self.layer_dims)
        self.activation_functions = activation_functions
        self.parameter_type = parameter_type
        self.parameters = dict()
        self.train_accuracy = list()
        self.val_accuracy = list()
        self.train_loss = list()
        self.history = dict()
        self.optimizer = optimizer
        
    def init_parameters(self):
        
        for i in range(1, self.L):
            
            self.parameters["b" + str(i)] = np.zeros((self.layer_dims[i],1))
            
            if self.parameter_type == "he":
                self.parameters["W" + str(i)] = np.random.randn(self.layer_dims[i],self.layer_dims[i-1]) * np.sqrt(2 / self.layer_dims[i-1])
                
            elif self.parameter_type == "xavier":
                self.parameters["W" + str(i)] = np.random.randn(self.layer_dims[i],self.layer_dims[i-1]) * np.sqrt(2 / (self.layer_dims[i] + self.layer_dims[i-1]))
                
            elif self.parameter_type == "default":
                self.parameters["W" + str(i)] = np.random.randn(self.layer_dims[i],self.layer_dims[i-1]) * 0.01
               
            
    def sigmoid_function(self, Z):
        res = 1 /(1 + (1 / np.exp(Z)))      
        return res
            
    def relu_function(self,Z):
        return np.maximum(0,Z)
    
    def softmax_function(self,Z):
        value = np.exp(Z - np.max(Z))
        res = value / np.sum(value,axis = 0)
        return res
    
    def sigmoid_derivative(self, Z):
        x = self.sigmoid_function(Z)
        return x * (1 - x)
    
    def softmax_derivative(self, Z):
        
        jacob_matrix = np.diag(Z)
        
        for i in range(len(jacob_matrix)):
            
            for j in range(len(jacob_matrix)):
                
                if i == j : 
                    jacob_matrix[i][j] = Z[i] * (1 - Z[i])
                
                else :
                    jacob_matrix[i][j] = -Z[i] * Z[j]
    
        return jacob_matrix
    
    def relu_derivative(self, Z):
        Z[Z <= 0] = 0
        Z[Z > 0] = 1
        return Z
    
    def forward_prop(self,X):
        
        cache = dict()
        
        cache["A" + str(0)] = X
        
        A = X
        
        for i in range(1, len(self.layer_dims)):
            Weight = self.parameters["W" + str(i)]
            bias = self.parameters["b" + str(i)]
            
            Z = np.dot(Weight, A) + bias
            
            if self.activation_functions[i-1] == "relu":
                A = self.relu_function(Z)
            
            elif self.activation_functions[i-1] == "softmax":
                A = self.softmax_function(Z)
            
            elif self.activation_functions[i-1] == "sigmoid":
                A = self.sigmoid_function(Z)
            
            cache["A" + str(i)] = A
            cache["Z" + str(i)] = Z
            
        AL = cache["A" + str(len(self.layer_dims) - 1)]
        
        return AL, cache
    
    def backward_prop(self, AL,cache, X, Y):
        
        grads = dict()
        
        dZ = AL - Y #derivative of the putput layer 
        
        for i in reversed(range(1, len(self.layer_dims))):
            
            if self.activation_functions[i-1] == "relu":
                dA = np.dot(self.parameters["W" + str(i+1)].T, dZ)
                dZ = dA * self.relu_derivative(cache["Z" + str(i)])

            elif self.activation_functions[i-1] == "sigmoid":
                dA = np.dot(self.parameters["W" + str(i+1)].T, dZ)
                dZ = dA * self.sigmoid_derivative(cache["Z" + str(i)])

            grads["dW" + str(i)] = np.dot(dZ, cache["A" + str(i-1)].T) / dZ.shape[1]
            grads["db" + str(i)] = np.sum(dZ,axis = 1, keepdims = True) / dZ.shape[1]
     
        return grads
    
    def find_cost(self,AL,Y):

        cost = - np.mean(Y * np.log(AL + 1e-8))
        
        return np.squeeze(cost)
    
    def predict(self,X,Y):
        
        AL, cache = self.forward_prop(X)
        
        A = np.argmax(AL, axis = 0)
        
        Y_pred = np.argmax(Y, axis = 0)
        
        return accuracy_score(A, Y_pred)*100
    
    def update_params(self,grads, learning_rate):
        
        for i in range(1,len(self.layer_dims)):
            self.parameters["W" + str(i)] = self.parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
            self.parameters["b" + str(i)] = self.parameters["b" + str(i)] - learning_rate * grads["db" + str(i)]
    
    def random_mini_batches(self, X, Y, mini_batch_size):
        m = X.shape[1]
        mini_batches = []
        
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            k = m // mini_batch_size
            mini_batch_X = shuffled_X[:, k*mini_batch_size :]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches
    
    def plot_cost(self,costs, epochs):
        ax = sns.lineplot(x =range(1,epochs + 1), y = costs)
        ax.set(xlabel='Epochs', ylabel='Cost')
        plt.show()
    
    def fit(self,X_train,Y_train,X_test,Y_test,learning_rate,epochs,batch_size):
        
        t = 0
        
        self.init_parameters()
        
        if self.optimizer:
            self.optimizer.init_params(self.layer_dims)

        costs = list()
            
        for i in range(1, epochs + 1):
            
            np.random.seed(i)
            mini_batches = self.random_mini_batches(X_train, Y_train, batch_size)
            
            cost = 0
            
            for minibatch in mini_batches:

                minibatch_X, mini_batch_Y = minibatch

                AL, cache = self.forward_prop(minibatch_X)

                cost += self.find_cost(AL, mini_batch_Y)
                                
                grads = self.backward_prop(AL, cache, minibatch_X, mini_batch_Y)

                if not self.optimizer:
                    self.update_params(grads,learning_rate)

                else:
                    
                    if self.optimizer.optimizer_name() == "Adam":
                        t = t + 1
                        self.parameters = self.optimizer.update_params_using_Adam(grads, t, self.layer_dims, self.parameters)

                    elif self.optimizer.optimizer_name() == "RMS":
                        self.parameters = self.optimizer.update_params_using_RmsProp(grads, self.layer_dims, self.parameters)            
                     
                    elif self.optimizer.optimizer_name() == "Adamax":
                        t = t + 1
                        self.parameters = self.optimizer.update_params_using_Adamax(grads, t, self.layer_dims, self.parameters)
                       
                    
            costs.append(cost/len(mini_batches))
            
            train_acc = self.predict(X_train,Y_train)
            
            val_acc = self.predict(X_test,Y_test)
            
            self.train_accuracy.append(train_acc)
            self.val_accuracy.append(val_acc)
                
            print("Epoch " + str(i) + " Train Accuracy == ",train_acc, "Test Accuracy == ",val_acc)

            
        self.train_loss = costs
        
        self.history["train_loss"] = self.train_loss
        self.history["train_accuracy"] = self.train_accuracy
        self.history["val_accuracy"] = self.val_accuracy
        
        return self.history
    
