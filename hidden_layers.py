import numpy as np

class HiddenLayer:

    # Size of all instance variables are dependant on the previous layer.
    def __init__(self, num_inputs, num_nodes):
        self.inputs = None 
        self.bias = np.ones(num_nodes)
        self.weights = np.random.random((num_nodes, num_inputs))
        self.layer_error = np.zeros(num_nodes)
        self.outputs = None 

        # Used for batch learning.
        self.batch_error = np.zeros(num_nodes)

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


    # Ei ← sigmoidError(Ci ) × [Di − Ci]
    # Pattern_Errori += (Di − Ci ) ** 2
    def calculate_output_error(self, desired):

        self.layer_error = self.sigmoid(self.outputs, derivative = True) * (desired - self.outputs)
        
        #print("*=========OUTPUT=========")
        #print("Output Error: " + str(self.layer_error))
        #print("Output: " + str(self.outputs) + " Output derivative : " + str(self.sigmoid(self.outputs, derivative = True)))
        #print("Desired: " + str(desired) + " desired - output: " + str((desired - self.outputs)))
        
        pattern_error = 0
        for node in range(0, len(self.outputs)):
            pattern_error += (desired[node] - self.outputs[node]) ** 2
        
        return pattern_error

    # Calculates the error of each hidden layer.
    # Ei = sigmoidError(Bi) * 􏰈 Sum WijEj
    # The object structure (the hidden layer contains the previous layers weights, bias).
    # For this reason HiddenLayer n+1 is used to calculate the layer error for HiddenLayer n.
    def calculate_layer_error(self):
    
        error = None
        #print("Weight: " + str(self.weights) + "\nError " + str(self.layer_error) + "\nInputs: " + str(self.inputs))

        if len(self.layer_error) == 1:
            
            error = self.sigmoid(self.inputs, derivative = True) * (self.weights * self.layer_error)
            #print("*=========LAYER==========")
            #print("Bi: " + str(self.inputs) + " Bi': " + str(self.sigmoid(self.inputs, derivative = True)))
            #print("Weights: " + str(self.weights))
            #print("Ej: " + str(self.layer_error))
            #print("Weights X Ej : " + str((self.weights * self.layer_error)))
            #print("Error: " + str(error))
           
        else:
            error = self.sigmoid(self.inputs, derivative = True) * (self.weights.T.dot(self.layer_error))


        # Ravel turns the error from a Nx1 matrix to a N sized vector. Vector form is needed in next step.
        return error.ravel()

    # Adjusting the weights based of layer error.
    def adjust_weights_and_bias(self, learning_rate, momentum):

        # Weight += αBiEj + αBiEjρ        
        change = np.outer(self.layer_error, self.inputs) * learning_rate

        #print("*=========ADUSTING WEIGHTS & BIAS'S=========")
        #print("Inputs: " + str(self.inputs))
        #print("layer Error: " + str(self.layer_error))
        #print("Input * Error: " + str(np.outer(self.layer_error, self.inputs)))
        #print("Learning Rate: " + str(learning_rate))
        #print("Deltas: " + str(change))
        #print("Current Weights: " + str(self.weights))
        #print("Adjusted Weights: " + str(self.weights + change))
        #print("Current bias:" + str(self.bias))
        #print("Learing Rate * Error: " + str((self.layer_error * learning_rate)))
        #print("Adjusted Bias: " + str(self.bias + (self.layer_error * learning_rate)))

        if momentum > 0:
            self.weights = self.weights + change + (change * momentum)
        else:
            self.weights = self.weights + change
 
        # Bias += αEi
        self.bias = self.bias + (self.layer_error * learning_rate)

    
    def batch_learning(self):
        pass

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



