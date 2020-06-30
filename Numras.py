#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Numras():
    
    def __init__(self, layer_dims, activation_functions, parameter_type = "default",optimizer = None):
        self.layer_dims = layer_dims
        self.L = len(self.layer_dims)
        self.activation_functions = activation_functions
        self.parameter_type = parameter_type
        self.parameters = dict()
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
        
        dZ = None
        
        for i in reversed(range(1, len(self.layer_dims))):
            
            if self.activation_functions[i-1] == "relu":
                dA = np.dot(self.parameters["W" + str(i+1)].T, dZ)
                dZ = dA * self.relu_derivative(cache["Z" + str(i)])

            elif self.activation_functions[i-1] == "sigmoid":
                dA = np.dot(self.parameters["W" + str(i+1)].T, dZ)
                dZ = dA * self.sigmoid_derivative(cache["Z" + str(i)])
            
            elif self.activation_functions[i-1] == "softmax":
                dZ = AL - Y
                
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
            
            print("Epoch " + str(i) + " Train Accuracy == ",self.predict(X,Y), "Test Accuracy == ",self.predict(X_test,Y_test))

        self.plot_cost(costs,epochs)


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import sklearn.datasets
import math
from numba import jit


# In[ ]:


class Adam():
    
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


# In[ ]:


class Adamax():
    
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


# In[ ]:


class RmsProp():
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


# In[ ]:


(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
enc = OneHotEncoder(sparse=False, categories='auto')

X = trainX.reshape(trainX.shape[0], trainX.shape[1]*trainX.shape[2]).T
Y = trainY.reshape(-1, trainY.shape[0])
Y = enc.fit_transform(Y.reshape(-1,1)).T
test_x = testX.reshape(testX.shape[0], testX.shape[1]*testX.shape[2]).T
test_y = testY.reshape(-1, testY.shape[0])
test_y = enc.transform(test_y.reshape(-1, 1)).T

X = X / 255
test_x = test_x / 255

print(f"X shape - {X.shape}")
print(f"Y shape - {Y.shape}")
print(f"test_x shape - {test_x.shape}")
print(f"test_y shape - {test_y.shape}")


# In[ ]:


layer_dims = [784,64,10]

activations = ["relu","softmax"]

opt = Adamax()

# opt = Adam()

# opt = RmsProp()

# opt = None

model = Numras(layer_dims,activations,parameter_type="he",optimizer=opt)

model.fit(X,Y,test_x,test_y,0.01,10,batch_size = 1024)


# In[ ]:


from keras.layers import Dense, Flatten

from keras.models import Sequential

from keras.initializers import glorot_normal, glorot_uniform, he_uniform, he_normal

from keras.optimizers import Adam as keras_adam

from keras.optimizers import Adamax as keras_adamax

from keras.optimizers import RMSprop as keras_Rms_prop

from keras.optimizers import Adagrad as keras_adagrad

from keras.optimizers import SGD as keras_sgd

model = Sequential()
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation = "softmax"))


opt1 = keras_adamax(learning_rate=0.001)

# opt1 = keras_Rms_prop(learning_rate=0.001)

model.compile(optimizer=opt1,loss="categorical_crossentropy",metrics=["accuracy"],initializers=he_normal())

history = model.fit(X.T,Y.T,validation_data=(test_x.T,test_y.T),epochs=10,batch_size = 1024)


# In[ ]:


y = history.history["loss"]

x = [i for i in range(1,11)]

fig, axs = plt.subplots(ncols=2)

ax = sns.lineplot(x = x, y = y,ax=axs[0])

ax.set(xlabel='Epochs', ylabel='Cost')

