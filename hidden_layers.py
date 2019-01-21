import numpy as np

class HiddenLayer:

    # Size of all instance variables are dependant on the previous layer.
    def __init__(self, num_inputs, num_nodes):
        self.inputs = np.zeros(num_inputs)
        self.bias = np.ones(num_nodes)
        self.weights = np.random.random((num_nodes, num_inputs))
        self.outputs = None #np.zeros(num_inputs)

    # Does the summation and puts it through the activation function.
    def calculate_output(self):
        
        # Calculating summation and adding bias for each node.
        self.outputs = self.weights.dot(self.inputs)
        self.outputs += self.bias

        # Putting them through Activation Function (sigmoid).
        for i in range(0, len(self.outputs)):
        	self.outputs[i] = self.sigmoid(self.outputs[i])

    # Calculating Sigmoid function.
    def sigmoid(self, x, derivative = False):
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm

    # Adding input to the layer before computation.
    def initialize_input(self, inputs):
    	self.inputs = inputs

    # Returning output of current layer.
    def get_output(self)
        return self.outputs
   
    def to_string(self):
        print("inputs: " + str(self.inputs))
        print("bias: " + str(self.bias))
        print("outputs: " + str(self.outputs))
        print("weights: " + str(self.weights))




#i = HiddenLayer(3, 4)
#i.testing()


