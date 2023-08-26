import numpy as np
import utils as utils

class Neuron:
    def __init__(self, nid, n_inputs=None, weights=None, bias=None, activation_function=None):
        """
        creates a neuron that can be defined uniquely through a nid (ny-dee)
        weights can be predefined via weights or randomly assigned via weights
        the bias can similarly be predefined or randomly assigned
        the activation function is sigmoid by default, others are relu and leaky relu
        """
        # assign nid
        self.nid = nid
        # define or assign weights
        self.weights = np.random.randn(n_inputs) if weights is None else np.array(weights)
        # define or assign bias
        self.bias = np.zeros(1) if bias is None else bias
        # define or assign activation function
        self.activation_function = 'sigmoid' if activation_function is None else activation_function
        # create trackers for processing
        self.z = None
        
    def process(self, inputs):
        self.z = np.clip(np.dot(inputs, self.weights) + self.bias, -100, 100)
        
    def activate(self):
        return utils.function_to_activation[self.activation_function](self.z)
        
    def derivative(self):
        return utils.function_to_derivative[self.activation_function](self.z)
    
    def clear(self):
        self.z = None