import numpy as np

# define helper calculation functions
# > loss = sum of squared errors
def calc_loss_sse(predicted, truth):
    return np.sum(np.power(predicted - truth, 2))

# define helper activation functions
# > sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return np.exp(x) / np.power(np.exp(x) + 1, 2)

# > relu
def relu(x):
    if x < 0:
        return 0
    return x
def relu_derivative(x):
    return 1 * (x >= 0)

# > leaky relu
def leaky_relu(x):
    if x < 0:
        return 0.01 * x
    return x
def leaky_relu_derivative(x):
    if x < 0:
        return 0.01
    return 1

# > function to activation map
function_to_activation = {'sigmoid': sigmoid,
                          'relu': relu,
                          'leaky_relu': leaky_relu}
def retrieve_f2a():
    return function_to_activation

# > function to derivative map
function_to_derivative = {'sigmoid': sigmoid_derivative,
                          'relu': relu_derivative,
                          'leaky_relu': leaky_relu_derivative}
def retrieve_f2d():
    return function_to_derivative