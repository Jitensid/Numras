# Numras

Numras is a basic Python Deep Learning API implemented in Numpy where one can develop simple neural nets just by passing Python lists containing information regarding activation functions and number of neurons in the layers of the model.It's results are comparable when same data is fed to a Keras model with the same Neural Network Architecture.

# Motivation for this project :

I read an article on Medium which stated that we should not treat ML and DL as "blackbox" and to get the best possible resulls one should understand what is going on inside the model and this motivated me to do this project.

# Details of Project :

One can even speed up gradient descent process by using advanced optimizers such as Adam, Adamax and RMSprop. If no optimizer is passed to the model then stochastic gradient descent is carried out by default. 

Currently this API supports 3 types of weight initializations - "defualt","he" and "xavier" initializations respectively with default weight initialization as "default". 

Also one can use mini-batch gradient descent to converge loss function quickly by passing the "batch_size" parameter in the fit function of the model.

Example : 

layer_dims = [784,64,10]

# list of number of neurons for the model layer by layer 
# 1st element is equal to shape of the Input

activations = ["relu","softmax"]
# list of activation functions for the model layer by layer 

opt = numras_adam()
# Initializing Adam optimizer from Numras

numras_model = Numras(layer_dims=layer_dims,activation_functions=activations,parameter_type="he",optimizer=opt)
# Initializing parameters of Weight and biases as he initialization 

numras_history = numras_model.fit(X,Y,test_x,test_y,0.01,10,batch_size = 1024)
# Fitting the model with the training data
