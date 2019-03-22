from hidden_layers import HiddenLayer
import numpy as np
import random
import sys


class NeuralNetwork:

    # Training stops when cumulative error is less than max.
    max_error = 0.01

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
    # learning_rate & momentum: the amount to which weights get adjusted by.
    # max_epoch: number of iterations through the training data set.
    # overfitting_data: array where index 0 is input data and index 1 is desired output. Used for overfitting check. 
    def train(self, data, desired, learning_rate, momentum, max_epoch, overfitting_data):
        
        # Error values are set to large values to begin with. 
        epoch = 0
        previous_epoch = 0
        cumulative_error = 999999
        previous_epoch_error = 999999

        # Used to prevent overfitting.
        previous_overfitting_error = 999999
        nn_prior_to_overfitting = self.hidden_layers

        # Randomize traversal order in each epoch.
        indexes = []
        for i in range(0, len(data)):
            indexes.append(i)

        # Train until you reach max num of epochs or error is below desired limit.
        while epoch < max_epoch and cumulative_error > self.max_error:
            
            cumulative_error = 0
            random.shuffle(indexes)

            # Forward & backward propagation.
            for i in indexes:
                self.forward_prop(data[i])
                cumulative_error += self.backward_prop(learning_rate, momentum, desired[i])

            # Scale error by number of samples.
            cumulative_error /= len(data)

            # Checking for correct trends every 5 epochs & updating learing rate.
            if epoch % 5 == 0:
                try:
                    # If the slope of the epoch error is positive that means the error is increasing. Training is stopped.
                    if float(cumulative_error - previous_epoch_error) / (epoch - previous_epoch) > 0.01:
                        print("Model Error is Rising at Epoch: " + str(epoch) + " Training is Stopped")
                        print(str(cumulative_error) + " - " + str(previous_epoch_error) +" / "+ str(epoch) +" - "+ str(previous_epoch))
                        return

                    # If validation accuracy decreases that means the program is overfitting the training data.
                    # Weights & biases are set to previous values and training is stopped.
                    # NOTE: Don't use the validation dataset to check for overfitting. Overfitting is only checked
                    # if dataset is large enough that the overfitting check has its own dataset.
                    overfitting_error = self.validation(overfitting_data[0], overfitting_data[1])
                   
                    if overfitting_error - previous_overfitting_error > 0.01 or (overfitting_error - cumulative_error > 0.05):
                        print("PRE OVERFITTING:")
                        self.to_string()
                        print("Model Overfitted the Training Set at Epoch: " + str(epoch) + " Training is Stopped")
                        print(str(overfitting_error) + " > " + str(previous_overfitting_error) + " Epoch: " + str(epoch))
                        self.hidden_layers = nn_prior_to_overfitting
                        print("POST OVERFITTING:")
                        self.to_string()
                        return

                    previous_overfitting_error = overfitting_error

                    # Updating learning rate based off the rate of change in epoch error.
                    if epoch > 0:
                        percent_change = (cumulative_error - previous_epoch_error) / (epoch - previous_epoch)
                        
                        # Learning rate should never increase. 
                        if percent_change < 0:
                            new_learning_rate = learning_rate - abs(learning_rate * percent_change)
    
                            if new_learning_rate > 0.01:
                                learning_rate = 0.01
                            elif new_learning_rate < 0.005:
                                learning_rate = 0.005
                            else:
                                learning_rate = new_learning_rate
                        
                except ZeroDivisionError:
                    pass

                # Updating values used for checks.
                previous_epoch_error = cumulative_error
                previous_epoch = epoch
                nn_prior_to_overfitting = self.hidden_layers

            # Prints Epoch error. 
            if epoch % 20 == 0:
                print(str(epoch) + ". ->Error Epoch: " + str(cumulative_error) + "Validation: " + str(self.validation(overfitting_data[0], overfitting_data[1])))

            epoch += 1

    # Description: Forward propagates and calculates output error without backpropagating the error.
    # It is used to check the accuracy of the model.
    # data & desired: Input data and desired output.
    def validation(self, data, desired):
        
        # Total Output error for the Epoch.
        cumulative_error = 0

        # Getting model output for input.
        for index in range(0, len(data)):
            self.forward_prop(data[index])
            cumulative_error += self.hidden_layers[-1].calculate_output_error(desired[index])
            #print("Desired: " + str(desired[index]) + " Output: " + str(self.hidden_layers[-1].get_output()) + " cumu: " + str(cumulative_error) + " += " + str(self.hidden_layers[-1].calculate_output_error(desired[index])))

        # Scaling error by number of samples.
        cumulative_error /= len(data)
        #print("validation error: " + str(cumulative_error))

        return cumulative_error



    # Description: Propagates output from first layer to the last to calculate output.
    # data: data passed into the first layer of network as input.
    def forward_prop(self, data):

        for hidden_layer in self.hidden_layers:

            # Initializing input data for layer to be evaluated.
            hidden_layer.initialize_input(data)
            hidden_layer.calculate_output()

            # Collecting evaluated output because it is the input to the next layer.
            data = hidden_layer.get_output()

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





























        