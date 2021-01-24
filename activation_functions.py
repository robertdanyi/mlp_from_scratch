# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
build a nn from scratch using numpy by Piotr Skalski
"""


import numpy as np



def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0,Z)


def sigmoid_derivative(dA, Z):
    """
    dA: the "error message" coming back from the next layer
    """
    sig = sigmoid(Z)
    return dA * sig * (1 - sig) # dA * dsig/dZ


def relu_derivative(dA, Z):
    """
    dA: the "error message" coming back from the next layer
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ



