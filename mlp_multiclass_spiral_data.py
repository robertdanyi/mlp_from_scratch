#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

toy multilayer perceptron for classifying a "spiral dataset"
reference:
    https://cs231n.github.io/neural-networks-case-study/
"""


import numpy as np
import matplotlib.pyplot as plt
import activation_functions as af
from tqdm import trange


N = 5000 # nr of examples per class => N*K rows
K = 3 # nr of classes
D = 2 # dimensionality of input


network = [
            {"units":D, "activation":None},
            {"units":100, "activation":"relu"},
            {"units":20, "activation":"relu"},
            {"units":K, "activation":None}] # we'll use softmax+cross-entropy for loss func

activations = {"relu":af.relu, "sigmoid":af.sigmoid}
de_activations = {"relu": af.relu_derivative, "sigmoid": af.sigmoid_derivative} 


def generate_dataset():
    """ generate spiral data for classification """
    
    X = np.zeros((N*K, D)) # N*K rows
    y = np.zeros(N*K, dtype='uint8')
    
    for k in range(K): # for each class index
        kx = range(N*k, N*(k+1)) # kx the k'th N row 
        r = np.linspace(0.0, 1, num=N) # growing radius values for each n in N
        theta = np.linspace(k*4, (k+1)*4, num=N) + np.random.randn(N)*0.2 # true theta with noise
        X[kx] = np.stack([r*np.sin(theta), r*np.cos(theta)], axis=1)
        y[kx] = k
        
    # visualise the ocean
    # plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    
    return X,y
    
    
def normalize_data(data): # not needed for the generated spiral data
    """ Center data at 0 with unit standard deviation """
    data = (data - np.mean(data)) / np.std(data)
    

def initialize_parameters():
    """
    W matrices will be of shape (n_input, n_output)
    We'll use Xavier (for ReLU) init for the hidden layers
    """    
    parameters = {}
    for l in range(1, len(network)-1):
        parameters["W"+str(l)] = np.random.normal(loc=0.0, 
                                                  scale=(np.sqrt(2/network[l-1]["units"])),
                                                  size=(network[l-1]["units"], network[l]["units"])) * 0.01
        parameters["b"+str(l)] = np.zeros((1, network[l]["units"]))
    
    l = len(network)-1 # output layer
    parameters["W"+str(l)] = np.random.randn(network[l-1]["units"], network[l]["units"]) * 0.01
    parameters["b"+str(l)] = np.zeros((1, network[l]["units"]))
    
    return parameters


def forward(X, network, parameters):
    
    cache = {}
    A_prev = X
    L = len(network)
    
    for l in range(1, L):
        needs_activation = network[l]["activation"]
        
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        Z = np.dot(A_prev, W) + b
        if needs_activation:
            activation = activations[needs_activation]
            A_curr = activation(Z) 
        else:
            A_curr = Z # last layer has no activation
        cache["Z"+str(l)] = Z
        cache["A"+str(l-1)] = A_prev
        
        A_prev = A_curr
    
    return A_curr, cache


def softmax_cross_entropy_loss(Z, y):
    """ SOFTMAX can only be used if the output is a vector (of values for each class) per datapoint """
    
    n = Z.shape[0]
    
    probs = np.exp(Z) / (np.sum(np.exp(Z), axis=1, keepdims=True)) # softmax
    correct_class_log_probs = -np.log(probs[range(n), y])
    
    loss = np.sum(correct_class_log_probs) / n
    
    return loss


def gradient_soft_xentropy(Z, y):
    """ Calculates gradient of loss function wrt the output scores """
    
    n = Z.shape[0]
    probs = np.exp(Z) / (np.sum(np.exp(Z), axis=1, keepdims=True))
    grad = probs
    grad[range(n), y] = grad[range(n), y] - 1   # decreasing the probability for the correct class by 1 
    grad = grad / n
    
    return grad


def backprop(output, y, network, parameters, cache, learn_rate=0.5):
    
    L = len(network)
    
    # output layer
    # output = cache["A"+str(L-1)] # final unactivated linear output
    W = parameters["W"+str(L-1)]
    grad = gradient_soft_xentropy(output, y)
    A_prev = cache["A"+str(L-2)] 
    dW = np.dot(A_prev.T, grad) # D*K =>  grad (output): n*K, input: n*D
    db = np.sum(grad, axis=0, keepdims=True)
    dA_prev = np.dot(grad, W.T)
    
    # update
    parameters["W"+str(L-1)] -= learn_rate * dW
    parameters["b"+str(L-1)] -= learn_rate * db
    
    
    for l in reversed(range(1,L-1)):
        Z = cache["Z"+str(l)]
        de_activation = de_activations[network[l]["activation"]]
        dZ = de_activation(dA_prev, Z)
        dW = np.dot(cache["A"+str(l-1)].T, dZ) # W: D*N, A_prev: n*D, Z: n*N
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, parameters["W"+str(l)].T)
        
        # update
        parameters["W"+str(l)] -= learn_rate * dW
        parameters["b"+str(l)] -= learn_rate * db
        
    return parameters
        
        
def train(X, y, network, iterations):
    
    losses = []
    parameters = initialize_parameters()
    
    for i in trange(iterations):
        output, cache = forward(X, network, parameters)
        loss = softmax_cross_entropy_loss(output, y)
        parameters = backprop(output, y, network, parameters, cache)
        
        losses.append(loss)
        
    return parameters
        
        
def evaluate_model(X, y, parameters):
    
    # predicted classes
    prediction, _ = forward(X, network, parameters)
    predicted_classes = np.argmax(prediction, axis=1)
    accuracy = np.mean(predicted_classes == y)
    
    print(f"\ntraining accuracy is {accuracy}")
    
    
def main():
    
    X,y = generate_dataset()
    normalize_data(X)
    trained_parameters = train(X,y, network, iterations=3000)
    
    evaluate_model(X, y, trained_parameters)
        
        
    
if __name__ == "__main__":
    main()