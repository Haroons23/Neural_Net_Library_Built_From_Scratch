import numpy as np
import random
from neural_network import NeuralNetwork

    # NOTE: A NN WITH 1 HIDDEN LAYER, 3 HIDDEN NODES, 100K EPOCHS, AND A LEARNING RATE OF .06 WILL SOLVE THE XOR PROBLEM.
    #       A NN WITH 1 HIDDEN LAYER, 6 HIDDEN NODES, 100K EPOCHS, AND A LEARNING RATE OF .06 WILL COME VERY CLOSE TO SOLVING THE IRIS PROBLEM.

def run_neural_network():

    data = k_fold_cross_validation()
     
    data_input = data[0]
    data_labels = data[1]

    n = NeuralNetwork(4, [6], 1)
    n.to_string()

    # Cross Validation
    for test_fold in range(len(data_input) - 1, - 1, - 1):
        for train_fold in range(0, len(data_input)):
            if train_fold != test_fold:
                n.train(data_input[train_fold], data_labels[train_fold], 0.05, 0)

        for test_case in range(0, len(data_input[test_fold])):
            print("TestFold: " + str(test_fold) + " Desired: " + str(data_labels[test_fold][test_case]) + " Result: " + str(n.forward_prop(data_input[test_fold][test_case])))

    #data = np.array([[1, 1],[1, 0],[0, 1],[0, 0]])
    #results = np.array([[0],[1],[1],[0]])
    #n.train(data, results, .08, 0)

    #print(str("11" ) + str(n.forward_prop(np.array([1,1]))))
    #print(str("10" ) + str(n.forward_prop(np.array([1,0]))))
    #print(str("01" ) + str(n.forward_prop(np.array([0,1]))))
    #print(str("00" ) + str(n.forward_prop(np.array([0,0]))))

    #print(str("1: 0.3 " ) + str(n.forward_prop(np.array([-0.55555556, 0.25,        -0.86440678,  -0.91666667  ]))))
    #print(str("2: 0.6 " ) + str(n.forward_prop(np.array([-0.66666667, -0.66666667, -0.22033898,  -0.25        ]))))
    #print(str("3: 0.9 " ) + str(n.forward_prop(np.array([-0.22222222, -0.33333333,  0.05084746,   0.          ]))))


def k_fold_cross_validation():

    # Getting data.
    data = read_training_data()

    # Each data fold will have 20 cases.
    fold_size = 20
    data_input = []
    data_label = []
 
    start = 0
    end = fold_size

    while start < len(data[0]):
        data_input.append(data[0][start:end])
        data_label.append(data[1][start:end])

        start += fold_size
        end += fold_size

    #for i in range(0, len(data_label)):
    #    for j in range(0, len(data_label[i])):
    #        print(str(data_input[i][j]) + ":" + str(data_label[i][j]))

    return data_input, data_label


# Reading in data from CSV file and storing it into a 2D array.
def read_training_data():

    healthy_cases = "AANEM-data/combined-data/healthy.csv"
    diseased_cases = "AANEM-data/combined-data/diseased.csv"
    iris = "../Iris Testing Data/bezdekIris-data.txt"

    data_shuffle = []

    # Opening data file.
    with open(iris, "r") as file_ptr:
        
        # Removing newline character and parsing.
        line_num = 0
        for line in file_ptr:
            line = line.rstrip()
            line = line.split(",")
            
            # Column headers. 
            if line_num == 0:
                line_num = 1
            
            # Data.  
            else:
                data_shuffle.append(line)
    
    # Shuffling data around.
    random.shuffle(data_shuffle)

    data_input = []
    data_labels = []

    # Breaking data into input and label.
    for row in data_shuffle:
        temp_input = []
        temp_label = []

        for i in range(0, len(row) - 1):
            temp_input.append(float(row[i]))
        
        temp_label.append(float(row[-1])) 

        data_input.append(temp_input)
        data_labels.append(temp_label)


    # Preprocessing and converting to numpy arrays.
    preprocessed_data = preprocess_data_0_1(data_input)
    labels = np.array(data_labels)

    return preprocessed_data, labels


# Preprocesses data into 2D numpy array with values between -1:1. Finds the max and min 
# of each variable, then uses it to perform transformation on data.
# x" = ((x - min) / ((max - min) / 2)) - 1.
def preprocess_data_0_1(data):

    # Converting list to a numpy array and getting max and min for each variable(axis 0 gets the columns max value). 
    data = np.array(data)
    mins = data.min(axis = 0)
    maxs = data.max(axis = 0)

    # Transforming data based off min and max.
    for variable in range(0, 4):
        min_val = mins[variable]
        max_val = maxs[variable]

        for row in range(0, len(data)):
            try:
                data[row][variable] = ((data[row][variable] - min_val) / ((max_val - min_val) / 2)) - 1 
            except ZeroDivisionError:
                data[row][variable] = 0

    return data

    


    '''
    # Alternate implementation without using NumPy.
    max_min_values = []

    # Each row reps an electronic signal with 7 variables of data.  
    for variable in range(0, 7):

        temp_min = data[0][variable]
        temp_max = data[0][variable]

        # Looking for min and max for each variable/column.
        for row in range(0, len(data)):
            if data[row][variable] < temp_min:
                temp_min = data[row][variable]
            if data[row][variable] > temp_max:
                temp_max = data[row][variable]

        max_min_values.append([temp_min, temp_max])
    

    # Transforming data based off min and max.
    for variable in range(0, 7):
        min_val = max_min_values[variable][0]
        max_val = max_min_values[variable][1]

        for row in range(0, len(data)):
            try:
                print(str(data[row][variable]))
                data[row][variable] = ((data[row][variable] - min_val) / ((max_val - min_val) / 2)) - 1 
                print(str(data[row][variable]))
            except ZeroDivisionError:
                data[row][variable] = 0
    '''
    


#np.random.seed(0)
run_neural_network()










