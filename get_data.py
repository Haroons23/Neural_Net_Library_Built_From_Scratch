import numpy as np
from neural_network import NeuralNetwork


def run_neural_network():
    
    # NOTE: A NN WITH 1 HIDDEN LAYER, 3 HIDDEN NODES, 100K EPOCHS, AND A LEARNING RATE OF .06 WILL SOLVE THE XOR PROBLEM.



    # Getting data in matrix form.
    data = read_training_data()
    data_input = data[0]
    data_labels = data[1]

    n = NeuralNetwork(4, [3], 1)
    n.to_string()

    n.train(data_input, data_labels, 0.05, 0)

    #data = np.array([[1, 1],[1, 0],[0, 1],[0, 0]])
    #results = np.array([[0],[1],[1],[0]])
    #n.train(data, results, .08, 0)

    #print(str("11" ) + str(n.forward_prop(np.array([1,1]))))
    #print(str("10" ) + str(n.forward_prop(np.array([1,0]))))
    #print(str("01" ) + str(n.forward_prop(np.array([0,1]))))
    #print(str("00" ) + str(n.forward_prop(np.array([0,0]))))

    print(str("1: 0.3 " ) + str(n.forward_prop(np.array([-0.55555556, 0.25, -0.86440678, -0.91666667]))))
    print(str("2: 0.6 " ) + str(n.forward_prop(np.array([-0.66666667, -0.66666667, -0.22033898, -0.25  ]))))
    print(str("3: 0.9 " ) + str(n.forward_prop(np.array([-0.22222222, -0.33333333,  0.05084746,  0. ]))))





# Reading in data from CSV file and storing it into a 2D array.
def read_training_data():

    healthy_cases = "AANEM-data/combined-data/healthy.csv"
    diseased_cases = "AANEM-data/combined-data/diseased.csv"
    iris = "../Iris Testing Data/bezdekIris-data.txt"
    data = []
    labels = []

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
                single_unit_data = []
                temp_label = []
                for i in range(0, len(line) - 1):
                    single_unit_data.append(float(line[i]))
                data.append(single_unit_data)

                # Labels
                temp_label.append(float(line[-1]))
                labels.append(temp_label)

    preprocessed_data = preprocess_data_0_1(data)
    labels = np.array(labels)

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










