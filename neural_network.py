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


    # Creates hidden layers. Each array index is one layer. Index value is number of nodes in that hidden layer.
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

    # The number of input going into the output layer is equal to the number of nodes in the last hidden layer.
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

    def train(self, data, desired, learning_rate, momentum):

        cumulative_error = 0.10
        epoch = 0

        # Used to randomize traversal order in each epoch.
        indexes = []
        for i in range(0, len(data)):
            indexes.append(i)

        # Train until you run out of data or error is below limit.
        while epoch < 1 and cumulative_error > self.max_error:
            
            cumulative_error = 0
            random.shuffle(indexes)

            for i in indexes:
                #print(str("Data to Input Layer: " + str(data[i])))
                self.forward_prop(data[i])
                cumulative_error += self.backward_prop(learning_rate, momentum, desired[i])

            # Scale error by number of samples.
            cumulative_error /= len(data)
            print(str(epoch) + ". Error for Epoch: " + str(cumulative_error))
            epoch += 1

    # Forward Propagation: Uses the output values of Layeri to calculate Layeri+1.
    def forward_prop(self, data):

        for hidden_layer in self.hidden_layers:

            # Initializing input data for layer to be evaluated.
            hidden_layer.initialize_input(data)
            hidden_layer.calculate_output()

            # Collecting evaluated output because it is the input to the next layer.
            data = hidden_layer.get_output()
            #hidden_layer.to_string()

        return (self.hidden_layers[-1].get_output())

    # Backward Propagation.
    def backward_prop(self, learning_rate, momentum, desired):
    
        # Calculating output error.
        #pattern_error = self.output_layer.calculate_output_error(desired)
        pattern_error = self.hidden_layers[-1].calculate_output_error(desired)
        #post_layer_error = self.hidden_layers[-1].get_error()

        # Calculating hidden layer errors. Starting at len() - 2 because len() - 1 is output layer. 
        for layer in range(len(self.hidden_layers) - 1, 0, - 1):
            error = self.hidden_layers[layer].calculate_layer_error()#(post_layer_error)
            self.hidden_layers[layer - 1].set_error(error)
            #post_layer_error = self.hidden_layers[layer].get_error()

        #print("\n\nTIME TO ADJUST ALL THE WEIGHTS!!!!\n\n")
        # Adjusting the weights for each layer. 
        for layer in self.hidden_layers:
            layer.adjust_weights_and_bias(learning_rate, momentum)
        
        return pattern_error


    def to_string(self):
        print("===========================")
        for i in self.hidden_layers:    
            i.to_string()
        print("===========================")


#np.random.seed(0)
n = NeuralNetwork(2, [2], 1)
n.to_string()

'''
and_training_data = np.random.random((1000, 2))
and_results = np.zeros((1000, 1))

for row in range(0, len(and_training_data)):
    for i in range(0, len(and_training_data[row])):
        if and_training_data[row][i] >= 0.50:
            and_training_data[row][i] = 1
        else:
            and_training_data[row][i] = 0

ii=0
jj=0
kk=0
ll=0

for row in range(0, len(and_training_data)):
    if and_training_data[row][0] == 1 and and_training_data[row][1] == 1:
        and_results[row] = [0]
        ii+=1
    elif and_training_data[row][0] == 0 and and_training_data[row][1] == 1:
        and_results[row] = [0]
        jj+=1
    elif and_training_data[row][0] == 1 and and_training_data[row][1] == 0:
        and_results[row] = [0]
        kk+=1
    else:
        and_results[row] = [1]
        ll+=1
'''
data = np.array([[1, 1],[1, 0],[0, 1],[0, 0]])
results = np.array([[0],[1],[1],[0]])

n.train(data, results, .05, 0)

#print("finished training: " + " i: " + str(ii) + " j: " + str(jj) + " k: " + str(kk) + " l: " + str(ll) )
#exit()

print(str("1-1: " ) + str(n.forward_prop(np.array([1,1]))))
print(str("1-0: " ) + str(n.forward_prop(np.array([1,0]))))
print(str("0-1: " ) + str(n.forward_prop(np.array([0,1]))))
print(str("0-0: " ) + str(n.forward_prop(np.array([0,0]))))

























        