#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple binary classifier MLP using only numpy.
References:
https://cs231n.github.io/neural-networks-case-study/
https://towardsdatascience.com/deep-neural-networks-from-scratch-in-python-451f07999373
https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
https://towardsdatascience.com/how-to-build-a-diy-deep-learning-framework-in-numpy-59b5b618f9b7

Xavier initialization:
https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
"""


import numpy as np
import matplotlib.pyplot as plt


nn_architecture = [
    {"layer_size":4, "activation":None}, # input layer
    {"layer_size":5, "activation": "relu"},
    {"layer_size":4, "activation": "relu"},
    {"layer_size":3, "activation": "relu"},
    {"layer_size":1, "activation": "sigmoid"} # output layer
    ]


def initialize_parameters(nn_architecture, seed=99):
    np.random.seed(seed)
    nr_of_layers = len(nn_architecture)
    parameters = {}
    
    # Xavier initialization for ReLU
    for l in range(1, nr_of_layers-1):
        parameters["W"+str(l)] = np.random.normal(size=(nn_architecture[l]["layer_size"], nn_architecture[l-1]["layer_size"]),
                                                  loc=0, scale=np.sqrt(2/nn_architecture[l-1]["layer_size"]) ) * 0.01
        parameters["b"+str(l)] = np.zeros(nn_architecture[l]["layer_size"], 1)
        
    # simple weights initialization for the final layer
    l = nr_of_layers-1
    parameters["W"+str(l)] = np.random.randn(nn_architecture[l]["layer_size"], nn_architecture[l-1]["layer_size"]) * 0.01
    parameters["b"+str(l)] = np.zeros(nn_architecture[l]["layer_size"], 1)

    return parameters


# used activation functions
def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def sigmoid_gradient(dA,Z):
    return dA * sigmoid(Z)*(1-sigmoid(Z))

def relu(Z):
    return np.maximum(0,Z)

def relu_gradient(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


activations = {"relu":relu, "sigmoid":sigmoid}
activation_gradients = {"relu": relu_gradient, "sigmoid": sigmoid_gradient}


def model_forward(X, parameters, nn_architecture):
    
    forward_cache = {}
    A = X # m*n
    nr_of_layers = len(nn_architecture)
    
    for l in range(1,nr_of_layers):
        A_prev = A
        W = parameters["W"+str(l)] # n*d
        b = parameters["b"+str(l)]
        activation = activations[nn_architecture[l]["activation"]]
        Z = np.dot(W, A_prev) + b
        A = activation(Z)
        forward_cache["Z"+str(l)] = Z
        forward_cache["A"+str(l-1)] = A_prev
        
    return A, forward_cache


def compute_loss(AL, y):
    """ binary cross entropy """
    
    # note we have column vectors as inputs and output as well
    # AL is of shape 1 * m
    # y is of shape 1 * m
    m = y.shape[1]
    logprobs = ( y * np.log(AL) + (1 - y) * np.log(1 - AL) )
    loss = (-1/m) *  np.sum(logprobs)
    loss = np.squeeze(loss) # makes shape (1,5) to (5,)
    
    return loss


def compute_grad_loss(y, AL):
    """ Derivative of the loss function wrt the output of the final layer"""
    
    y.reshape(AL.shape) # shouldn't matter
    return -( (y/AL) - (1 - y)/(1 - AL))
    

def model_backward(y, AL, parameters, forward_cache, nn_architecture, learn_rate):
    
    nr_of_layers = len(nn_architecture)
    m = y.shape[1]
    dLdAL = compute_grad_loss(y, AL) # gradient of loss wrt to final output
    dA_prev = dLdAL
    
    for l in reversed(range(1, nr_of_layers)):
        
        activation_grad_func = activation_gradients[nn_architecture[l]["activation"]]
        
        dZ = activation_grad_func(dA_prev, forward_cache["Z"+str(l)]) # global gradient wrt linear output; shape: n*m, where m is nr of examples
        dW = np.dot(dZ, forward_cache["A"+str(l-1)].T) / m            # global gradient wrt the weights; dW shape must be (n(l),n(l-1))
        db = np.sum(dZ, axis=1, keepdim=True) / m                     # global gradient wrt the bias term
        dA = np.dot(parameters["W"+str(l)].T, dZ)                     # global gradient wrt the current layer's input; W: n*d; dZ: n*m, 
        
        # update weights and bias
        parameters["W"+str(l)] -= learn_rate * dW
        parameters["b"+str(l)] -= learn_rate * db
        
        dA_prev = dA
        
        return parameters
    
    
def predict(X, parameters):
    
    prediction = model_forward(X, parameters, nn_architecture)
    
    return prediction


def train(iterations, X, y):
    """ 
    X should be of shape (n*m) where 
        n is the nr of features
        m is the nr of data points
    """
    
    parameters = initialize_parameters
    learn_rate = 0.001
    losses = []
    
    for i in range(iterations):
        output, forward_cache = model_forward(X, parameters, nn_architecture)
        loss = compute_loss(output, y)
        parameters = model_backward(y, output, parameters, forward_cache, nn_architecture, learn_rate)
        losses.append(loss)
        
        if i%100 == 0:
            print(f'loss after {i} iterations:', loss)
            
    # plot the losses
    plt.plot(losses)
    plt.ylabel( 'loss')
    plt.xlabel('nr of iterations')
    plt.title(f'learning rate = {learn_rate}')
    plt.show()
        
    



