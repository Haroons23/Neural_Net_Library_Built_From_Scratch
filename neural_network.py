from hidden_layers import HiddenLayer
import numpy as np
import random
import sys


class NeuralNetwork:

    # Training stops when total error is less than max.
    max_error = 0.00005
    ONLINE_LEARNING = True

    def __init__(self, num_inputs, num_hidden_layers, num_outputs):
        self.num_inputs  = num_inputs
        self.hidden_layers = []
        self.output_layer = None
        self.create_hidden_layers(num_inputs, num_hidden_layers)
        self.create_output_layer(num_inputs, num_hidden_layers[-1], num_outputs)


    # Description: Creates a list of hiddenlayer objects which represent the hidden layers of the network.
    # num_inputs: number of inputs going into first hidden layer.
    # hidden_layers_info: an int array with each index representing one hidden layer. The value in index
    # represents number of nodes in the hidden layer. 
    def create_hidden_layers(self, num_inputs, hidden_layers_info):

        # Means Perceptron therefore no hidden layer as input goes straight to output node.
        if hidden_layers_info[0] == 0 and len(hidden_layers_info) == 1:
            return
       
        # A hidden layer must have alteast 2 nodes.
        for num_nodes in hidden_layers_info:
            assert(num_nodes > 1), "Hidden Layers Must Have More Than One Node!"

        # Create first layer.
        first_hl = HiddenLayer(num_inputs, hidden_layers_info[0])
        self.hidden_layers.append(first_hl)

        # Adding remaining hidden layers.
        for layer_num in range(1, len(hidden_layers_info)):
            temp_layer = HiddenLayer(hidden_layers_info[layer_num - 1], hidden_layers_info[layer_num])
            self.hidden_layers.append(temp_layer)

    # Description: Adds an output layer to the list containing the hidden layers.
    # num_inputs: if NN is a perceptron num inputs is used because inputs go straight to the output layer.
    # last_hidden_layer: number of nodes in last hidden layer is the number of inputs to the output layer.
    def create_output_layer(self, num_inputs, last_hidden_layer, num_outputs):

        # Checking for invalid dimensions.
        assert(num_inputs > 0 and num_outputs > 0), "NN Must Have Atleast One Output and Input!"

        # Creates a perceptron where input goes straight to output node. No hidden layer exists.
        if last_hidden_layer == 0:
            self.output_layer = HiddenLayer(num_inputs, num_outputs)

        # When a hidden layer exists.
        else:
            self.output_layer = HiddenLayer(last_hidden_layer, num_outputs)

        self.hidden_layers.append(self.output_layer)

    # Description: Runs the training simulation to modify the weights to their correct values.
    # data & desired: data to train with and the desired output of of each training case.
    # learning_rate & momentum: the amount to which weights can change by. 
    def train(self, data, desired, learning_rate, momentum):

        cumulative_error = 0.05
        epoch = 0

        # Used to randomize traversal order in each epoch.
        indexes = []
        for i in range(0, len(data)):
            indexes.append(i)

        # Train until you run out of data or error is below limit.
        while epoch < 100000 and cumulative_error > self.max_error:
            
            cumulative_error = 0
            random.shuffle(indexes)

            for i in indexes:
                #print("$$$$$$$$$" + str(epoch) + str(". Data to Input Layer: " + str(data[i])))
                self.forward_prop(data[i])
                cumulative_error += self.backward_prop(learning_rate, momentum, desired[i])

            # Scale error by number of samples.
            cumulative_error /= len(data)
            if epoch % 100 == 0:
                print(str(epoch) + ". ->Error for Epoch: " + str(cumulative_error))
                print(str("11" ) + str(self.forward_prop(np.array([1,1]))))
                print(str("10" ) + str(self.forward_prop(np.array([1,0]))))
                print(str("01" ) + str(self.forward_prop(np.array([0,1]))))
                print(str("00" ) + str(self.forward_prop(np.array([0,0]))))
            epoch += 1

    # Description: Propagates output from first layer to the last to calculate output.
    # data: data passed into the first layer of network as input.
    def forward_prop(self, data):

        for hidden_layer in self.hidden_layers:

            # Initializing input data for layer to be evaluated.
            hidden_layer.initialize_input(data)
            hidden_layer.calculate_output()

            # Collecting evaluated output because it is the input to the next layer.
            data = hidden_layer.get_output()
            #self.to_string()

        # Returning the last layers output also know as the NN's output.
        return (self.hidden_layers[-1].get_output())

    # Description: Propagates layer error values backwards through each layer to calculate changes in weights.
    # leanning_rate & momentum: the amount to which weights can change by. 
    # desired: the desired result of each training case, used to calculate output error.
    def backward_prop(self, learning_rate, momentum, desired):
    
        # Calculating output error.
        pattern_error = self.hidden_layers[-1].calculate_output_error(desired)

        # Calculating hidden layer errors. Starting at len() because len() - 1 is output layer. 
        for layer in range(len(self.hidden_layers) - 1, 0, - 1):
            error = self.hidden_layers[layer].calculate_layer_error()
            self.hidden_layers[layer - 1].set_error(error)

        # Adjusting the weights for each layer. 
        for layer in self.hidden_layers:
            layer.adjust_weights_and_bias(learning_rate, momentum)
        
        return pattern_error


    def to_string(self):
        print("===========================")
        for i in self.hidden_layers:    
            i.to_string()
        print("===========================")





























        