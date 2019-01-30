from hidden_layers import HiddenLayer
import numpy as np


class NeuralNetwork:

    # Training stops when total error is less than max.
    max_error = 0.05
    ONLINE_LEARNING = True

    def __init__(self, num_inputs, num_hidden_layers, num_outputs):
        self.num_inputs  = num_inputs
        self.hidden_layers = []
        self.output_layer = None
        self.create_hidden_layers(num_inputs, num_hidden_layers)
        self.create_output_layer(num_inputs, num_hidden_layers[-1], num_outputs)


    # Creates hidden layers. Each array index is one layer. Index value is number of nodes in that hidden layer.
    def create_hidden_layers(self, num_inputs, hidden_layers_info):

        # Creates a perceptron where input goes straight to output node. No hidden layer exists.
        if hidden_layers_info[0] == 0:
            return

        # Create first layer.
        first_hl = HiddenLayer(num_inputs, hidden_layers_info[0])
        self.hidden_layers.append(first_hl)

        # Adding remaining hidden layers.
        for layer_num in range(1, len(hidden_layers_info)):
            temp_layer = HiddenLayer(hidden_layers_info[layer_num - 1], hidden_layers_info[layer_num])
            self.hidden_layers.append(temp_layer)

    # The number of input going into the output layer is equal to the number of nodes in the last hidden layer.
    def create_output_layer(self, num_inputs, last_hidden_layer, num_outputs):

        # Creates a perceptron where input goes straight to output node. No hidden layer exists.
        if last_hidden_layer == 0:
            self.output_layer = HiddenLayer(num_inputs, num_outputs)

        # When a hidden layer exists.
        else:
            self.output_layer = HiddenLayer(last_hidden_layer, num_outputs)

        self.hidden_layers.append(self.output_layer)

    def train(self, data, learning_rate, momentum):

        cumulative_error = 1000000
        epoch = 0
        data = np.array([[.10, .20], [.30, .40]])
        #desired = np.array([[1, 1], [1, 1]])
        desired = np.array([[1], [1]])

        # Train until you run out of data or error is below limit.
        while epoch < 1 and cumulative_error > self.max_error:
            
            for i in range(0, len(data)):
                #print(str("Data to Input Layer: " + str(data[i])))
                self.forward_prop(data[i])
                cumulative_error += self.backward_prop(learning_rate, momentum, desired[i])
                    


            epoch += 1


    # Forward Propagation. The input and output layers are stored in the hidden layer.
    def forward_prop(self, data):

        for hidden_layer in self.hidden_layers:

            # Initializing input data for layer to be evaluated.
            hidden_layer.initialize_input(data)
            hidden_layer.calculate_output()

            # Collecting evaluated output because it is the input to the next layer.
            data = hidden_layer.get_output()
            #hidden_layer.to_string()

    # Backward Propagation.
    def backward_prop(self, learning_rate, momentum, desired):
    
        # Calculating output error.
        pattern_error = self.output_layer.calculate_output_error(desired)
        post_layer_error = self.output_layer.get_error()

        # Calculating hidden layer errors.
        for layer in range(len(self.hidden_layers) - 1, -1, -1):
            self.hidden_layers[layer].calculate_layer_error(post_layer_error)
            post_layer_error = self.hidden_layers[layer].get_error()

        # Adjusting the weights for each layer. 
        for layer in self.hidden_layers:
            layer.adjust_weights_and_bias(learning_rate, momentum)
        
        return pattern_error


    def to_string(self):
        for i in self.hidden_layers:
            i.to_string()
        print("\n")
        self.output_layer.to_string()


n = NeuralNetwork(2, [2, 2], 1)
#n.forward_prop()
n.train(1, .05, .10)


        