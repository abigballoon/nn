import numpy as np
import random

def sigmoid(z):
    result = float(1) / (float(1) + np.exp(-z))
    return result

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    result = sigmoid(z) * (1-sigmoid(z))
    return result

def mse_derivative(output, sample):
    result = output - sample
    return result

def split(data, percent):
    random.shuffle(data)
    index = int(len(data) * (1 - percent))
    return data[: index], data[index: ]

class EMPTY(object):
    """
    empty value
    """
    pass