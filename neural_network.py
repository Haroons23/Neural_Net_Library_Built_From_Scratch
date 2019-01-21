from hidden_layers import HiddenLayer
import numpy as np


class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden_layers, num_outputs):
        self.num_inputs  = num_inputs
        self.hidden_layers = []
        self.output_layer = None
        self.create_hidden_layers(num_inputs, num_hidden_layers)
        self.create_output_layer(num_hidden_layers[-1], num_outputs)


    # Creates hidden layers. Each array index is one layer. Index value is number of nodes in that hidden layer.
    def create_hidden_layers(self, num_inputs, hidden_layers_info):

        # Create first layer.
        first_hl = HiddenLayer(num_inputs, hidden_layers_info[0])
        self.hidden_layers.append(first_hl)

        # Adding remaining hidden layers.
        for layer_num in range(1, len(hidden_layers_info)):
            temp_layer = HiddenLayer(hidden_layers_info[layer_num - 1], hidden_layers_info[layer_num])
            self.hidden_layers.append(temp_layer)

    # The number of input going into the output layer is the number of nodes in the last hidden layer.
    def create_output_layer(self, last_hidden_layer, num_outputs):
        self.output_layer = HiddenLayer(last_hidden_layer, num_outputs)

    def forward_prop(self):

        test_data = np.array([10, 20])

        self.hidden_layers[0].initialize_input(test_data)
        self.hidden_layers[0].calculate_output()
        self.hidden_layers[0].to_string()






    def to_string(self):
        for i in self.hidden_layers:
            i.to_string()
 
        print("\n")
        self.output_layer.to_string()


n = NeuralNetwork(2, [3, 2], 1)
n.forward_prop()
#n.to_string()


        