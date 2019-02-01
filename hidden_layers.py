import numpy as np

class HiddenLayer:

    # Size of all instance variables are dependant on the previous layer.
    def __init__(self, num_inputs, num_nodes):
        self.inputs = None 
        self.bias = np.ones(num_nodes)
        self.weights = np.random.random((num_nodes, num_inputs))
        self.layer_error = np.zeros(num_nodes)
        self.outputs = None 

    # Does the summation and puts it through the activation function.
    def calculate_output(self):
        
        # Calculating summation and adding bias for each node.
        self.outputs = self.weights.dot(self.inputs)
        self.outputs += self.bias

        # Activation Function (sigmoid).
        for i in range(0, len(self.outputs)):
        	self.outputs[i] = self.sigmoid(self.outputs[i])

    # Calculating Sigmoid function.
    def sigmoid(self, x, derivative = False):

        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm


    # Ei ← sigmoidError(Ci ) × [Di − Ci ]
    # Pattern_Errori += (Di − Ci ) ** 2
    def calculate_output_error(self, desired):

        self.layer_error = self.sigmoid(self.outputs, derivative = True) * (desired - self.outputs)
        print("OUTPUT ERROR: " + str(self.layer_error) + " = " + str(self.sigmoid(self.outputs, derivative = True)) + " * (" + str(desired) + " - " + str(self.outputs) )

        pattern_error = 0
        for node in range(0, len(self.outputs)):
            pattern_error += (desired[node] - self.outputs[node]) ** 2
            #print("PATTERN ERROR: " + str(pattern_error) + " += (" + str(desired[node]) + " - " + str(self.outputs[node]) + ")^2")
        
        print("inputs: " + str(self.inputs) + " output: " + str(self.outputs) + " Desired: " + str(desired) +  " Pattern Error " + str(pattern_error))
   

        return pattern_error

    # Calculates the error of each hidden layer.
    # Ei = sigmoidError(Bi) * 􏰈 Sum WijEj
    # The object structure (the hidden layer contains the previous layers weights, bias).
    # For this reason HiddenLayer n+1 is used to calculate the layer error for HiddenLayer n.
    def calculate_layer_error(self):
    
        error = None

        if len(self.layer_error) == 1:
            error = self.sigmoid(self.inputs, derivative = True) * (self.weights * self.layer_error).sum()
            print("Solving HiddenLayer Error (1 Node): " + str(error) + " = " + str(self.sigmoid(self.inputs, derivative = True)) + " * " + str((self.weights * self.layer_error).sum()))
        else:
            error = self.sigmoid(self.inputs, derivative = True) * self.weights.dot(self.layer_error).sum()
            print("Solving HiddenLayer Error (2+ Nodes): " + str(error) + " = " + str(self.sigmoid(self.inputs, derivative = True)) + " * " + str(self.weights.dot(self.layer_error).sum()))

        return error

    # Adjusting the weights based of layer error.
    def adjust_weights_and_bias(self, learning_rate, momentum):

        #print("PRE ADJUSTMENT:")
        #self.to_string()
        # Weight += αBiEj + αBiEjρ
        # T is used to transpose the matrix so right columns and rows are added and then transposed back into regular positions.
         
        #print("Inputs: " + str(self.inputs) + " ERROR: " + str(self.layer_error))

        if momentum > 0:
            self.weights = (self.weights.T + ((self.layer_error * self.inputs * learning_rate) + (self.layer_error * self.inputs * learning_rate * momentum))).T
            print("change in weights = " + str((self.layer_error * self.inputs * learning_rate)) + " + " + str((self.layer_error * self.inputs * learning_rate * momentum)))
        else:
            #self.weights = (self.weights.T + ((self.layer_error * self.inputs * learning_rate))).T
            self.weights = (self.weights + ((self.layer_error * self.inputs * learning_rate)))
            print("change in weights = " + str((self.layer_error * self.inputs * learning_rate)))

        print("WEIGHTS: " + str(self.weights))

        #print("ADJUSTING ERROR: Weights" + str(self.weights))
        #self.to_string()

        #print("POST ADJUSTMENT:")
        #self.to_string()

        # Bias += αEi
        #print("ERROR: " + str(self.layer_error) + " OUTPUTS: " + str(self.inputs))
        self.bias += self.layer_error * learning_rate
        print("NEW BIAS: " + str(self.bias) + " += " + str(self.layer_error * learning_rate))
        #print(" adjusted weights OUTPUTS: " + str(self.outputs))

    # Adding input to the layer before computation.
    def initialize_input(self, inputs):
    	self.inputs = inputs

    # Returning output of current layer.
    def get_output(self):
        return self.outputs

    # Returning error of current layer.
    def get_error(self):
        return self.layer_error

    # Sets value for layer errors
    def set_error(self, error):
        self.layer_error = error
   
    def to_string(self):
        print("inputs: " + str(self.inputs))
        print("bias: " + str(self.bias))
        print("outputs: " + str(self.outputs))
        print("weights: " + str(self.weights))
        print("error: " + str(self.layer_error))
        print("-------------")



