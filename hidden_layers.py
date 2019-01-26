import numpy as np

class HiddenLayer:

    # Size of all instance variables are dependant on the previous layer.
    def __init__(self, num_inputs, num_nodes):
        self.inputs = None #np.zeros(num_inputs)
        self.bias = np.ones(num_nodes)
        self.weights = np.random.random((num_nodes, num_inputs))
        self.layer_error = np.zeros(num_nodes)
        self.outputs = None #np.zeros(num_inputs)

    # Does the summation and puts it through the activation function.
    def calculate_output(self):
        
        # Calculating summation and adding bias for each node.
        self.outputs = self.weights.dot(self.inputs)
        self.outputs += self.bias

        print("PRE Activation: " + str(self.outputs))

        # Putting them through Activation Function (sigmoid).
        for i in range(0, len(self.outputs)):
        	self.outputs[i] = self.sigmoid(self.outputs[i])

        print("POST Activation: " + str(self.outputs))

    # Calculating Sigmoid function.
    def sigmoid(self, x, derivative = False):

        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm


    # Output Error = sigmoidError(Ci) * (Di - Ci)
    # Pattern Error = Sum of (Di - Ci) ** 2
    def calculate_output_error(self, desired):
        self.layer_error = self.sigmoid(self.outputs, derivative = True) * (desired - self.outputs)
        print(str(self.layer_error) + " = " + str(self.sigmoid(self.outputs, derivative = True)) + " * (" + str(desired) + " - " + str(self.outputs) + ")")
        print("LAYER ERROR: " + str(self.layer_error))
        print("DESIRED: " + str(desired))
        print("COST: " + str(self.outputs))

        pattern_error = 0
        for node in range(0, len(self.outputs)):
            pattern_error += (desired[node] - self.outputs[node]) ** 2
            print(str(pattern_error) + " += (" + str(desired[node]) + " - " + str(self.outputs[node]) + ")^2")

        return pattern_error

    # Calculates the error of each hidden layer.
    # ERRORi = sigmoidError(Bi) * Sum of WijEj
    def calculate_layer_error(self, post_layer_error):
    

        ## Error with dot product num cols in first matrix != # rows in second matrix ##
        ## Fix this issue.

        self.layer_error = self.sigmoid(self.outputs, derivative = True) * self.weights.dot(post_layer_error).sum()
        print("Solving HiddenLayer Error: " + str(self.layer_error) + " = " + str(self.sigmoid(self.outputs, derivative = True)) + " * " + str(self.weights.dot(post_layer_error).sum()))
        print(">>>>>> " + str(self.layer_error))

    # Adding input to the layer before computation.
    def initialize_input(self, inputs):
    	self.inputs = inputs

    # Returning output of current layer.
    def get_output(self):
        return self.outputs

    # Returning error of current layer.
    def get_error(self):
        return self.layer_error
   
    def to_string(self):
        print("inputs: " + str(self.inputs))
        print("bias: " + str(self.bias))
        print("outputs: " + str(self.outputs))
        print("weights: " + str(self.weights))
        print("error: " + str(self.layer_error))




#i = HiddenLayer(3, 4)
#sigm = i.sigmoid(np.array([1,2,3,4,5,6]))
#sigm = i.sigmoid(24)
#print(str(sigm))


