import numpy as np
import random
from neural_network import NeuralNetwork

    # NOTE: A NN WITH 1 HIDDEN LAYER, 3 HIDDEN NODES, 100K EPOCHS, AND A LEARNING RATE OF .06 WILL SOLVE THE XOR PROBLEM.
    #       A NN WITH 1 HIDDEN LAYER, 6 HIDDEN NODES, 5K EPOCHS, AND A LEARNING RATE OF .05 WILL COME VERY CLOSE TO SOLVING THE IRIS PROBLEM.
    #       CONFIGURATIONS FOR IRIS hidden_layer_structure = [[5], [10], [10,10], [10,5], [5,10], [8,8,8], [10,8,6], [6,8,10], [6,6,6,6], [10,9,8,6]]

    # Good results: 10 nodes LR:0.07 

def run_neural_network_tune_and_train():
    
    data = k_fold_cross_validation()
     
    data_input = data[0]
    data_labels = data[1]

    hidden_layer_structure = [[8],[10],[8,4], [9],[2], [5,5],[10,5],[9],[20],[25],[15,10], [20,20], [20,15],[15],[5],[6],[2]]#,[4],[7],[8],[11], [12]]
    learning_rate = 0.03
    config = 0
    average_model_error = 0

    # Cross Validation: Concatenating the folds to make one training set.
    for test_fold in range(len(data_input) - 1, - 1, - 1):

        # Selecting validation fold. Validation fold is always the fold before the test fold.
        # If test fold is fold zero, the validation fold is the last fold.
        validation_fold = 0
        if test_fold == 0:
            validation_fold = len(data_input) - 1
        else:
            validation_fold = test_fold - 1

        combined_folds_input = []
        combined_folds_label = []

        # Creating Training Data Set.
        for train_fold in range(0, len(data_input)):
            # Not concatenating validation and test fold.
            if train_fold != test_fold and train_fold != validation_fold:
                # If training set is empty, set it equal to the first fold.
                if len(combined_folds_input) == 0:
                    combined_folds_input = data_input[train_fold]
                    combined_folds_label = data_labels[train_fold]
                # If training set is not empty, concatenate other test folds to it.
                else:
                    combined_folds_input = np.concatenate((combined_folds_input, data_input[train_fold]), axis = 0)
                    combined_folds_label = np.concatenate((combined_folds_label, data_labels[train_fold]), axis = 0)

        # Additional information.
        print("Hidden Layer(s): " + str(hidden_layer_structure[config]) + " LR: " + str(learning_rate))
        
        # Creating and training the model on the training data.
        n = NeuralNetwork(7, hidden_layer_structure[config], 7)
        n.train(combined_folds_input, combined_folds_label, learning_rate, 0, [data_input[validation_fold], data_labels[validation_fold]])
        
        # Validating model on validation set and then adjusting the hyperparameters.
        validation_error = n.validation(data_input[validation_fold], data_labels[validation_fold])
        print("--------- Validation Error: " + str(validation_error))
        #hidden_layer_structure[0] += 
        #hidden_layer_structure[1] += 1
        #learning_rate += .01
        config +=1

        correct = 0
        wrong = 0

        '''
        for test_case in range(0, len(data_input[test_fold])):
            temp_output = n.forward_prop(data_input[test_fold][test_case])
            #print("Output: " + str(temp_output))

            # Output analysis for EMG classification.
            if temp_output[0] > temp_output[1] and temp_output[0] > temp_output[2]: # Healthy
                final_output = 0
            elif temp_output[1] > temp_output[0] and temp_output[1] > temp_output[2]: # Myogenic
                final_output = 1
            elif temp_output[2] > temp_output[0] and temp_output[2] > temp_output[1]: # Neurogenic
                final_output = 2
            else:
                print("equal outputs!")

            if (data_labels[test_fold][test_case][0] == 1 and final_output == 0) or \
               (data_labels[test_fold][test_case][1] == 1 and final_output == 1) or \
               (data_labels[test_fold][test_case][2] == 1 and final_output == 2):
                correct += 1
            else:
                wrong += 1
        '''
            

        for test_case in range(0, len(data_input[test_fold])):
            temp_output = n.forward_prop(data_input[test_fold][test_case])

            if temp_output[0] > temp_output[1] and temp_output[0] > temp_output[2]and temp_output[0] > temp_output[3]and temp_output[0] > temp_output[4]and temp_output[0] > temp_output[5]and temp_output[0] > temp_output[6]: # Healthy
                final_output = 0
            elif temp_output[1] > temp_output[0] and temp_output[1] > temp_output[2]and temp_output[1] > temp_output[3]and temp_output[1] > temp_output[4]and temp_output[1] > temp_output[5]and temp_output[1] > temp_output[6]:
                final_output = 1
            elif temp_output[2] > temp_output[0] and temp_output[2] > temp_output[1]and temp_output[2] > temp_output[3]and temp_output[2] > temp_output[4]and temp_output[2] > temp_output[5]and temp_output[2] > temp_output[6]:
                final_output = 2
            elif temp_output[3] > temp_output[0] and temp_output[3] > temp_output[1]and temp_output[3] > temp_output[2]and temp_output[3] > temp_output[4]and temp_output[3] > temp_output[5]and temp_output[3] > temp_output[6]:
                final_output = 3
            elif temp_output[4] > temp_output[0] and temp_output[4] > temp_output[1]and temp_output[4] > temp_output[2]and temp_output[4] > temp_output[3]and temp_output[4] > temp_output[5]and temp_output[4] > temp_output[6]:
                final_output = 4
            elif temp_output[5] > temp_output[0] and temp_output[5] > temp_output[1]and temp_output[5] > temp_output[2]and temp_output[5] > temp_output[3]and temp_output[5] > temp_output[4]and temp_output[5] > temp_output[6]: # Neurogenic
                final_output = 5
            elif temp_output[6] > temp_output[0] and temp_output[6] > temp_output[1]and temp_output[6] > temp_output[2]and temp_output[6] > temp_output[3]and temp_output[6] > temp_output[4]and temp_output[6] > temp_output[5]: # Neurogenic
                final_output = 6
            else:
                print("equal outputs!")

            if (data_labels[test_fold][test_case][0] == 1 and final_output == 0) or \
               (data_labels[test_fold][test_case][1] == 1 and final_output == 1) or \
               (data_labels[test_fold][test_case][2] == 1 and final_output == 2) or \
               (data_labels[test_fold][test_case][3] == 1 and final_output == 3) or \
               (data_labels[test_fold][test_case][4] == 1 and final_output == 4) or \
               (data_labels[test_fold][test_case][5] == 1 and final_output == 5) or \
               (data_labels[test_fold][test_case][6] == 1 and final_output == 6):
                correct += 1
            else:
                wrong += 1

    
            '''
            # Output analysis for IRIS & Abalone classification.
            if temp_output[0] > temp_output[1] and temp_output[0] > temp_output[2]:
                final_output = 1
            elif temp_output[1] > temp_output[0] and temp_output[1] > temp_output[2]:
                final_output = 2
            elif temp_output[2] > temp_output[0] and temp_output[2] > temp_output[1]:
                final_output = 3
            else:
               final_output = "equal outputs"

            if (data_labels[test_fold][test_case][0] == 1 and final_output == 1) or (data_labels[test_fold][test_case][1] == 1 and final_output == 2) or (data_labels[test_fold][test_case][2] == 1 and final_output == 3):
                correct += 1
            else:
                wrong += 1 
            '''
            
            

            print("TestFold: " + str(test_fold) + " Desired: " + str(data_labels[test_fold][test_case]) + " Result: " + str(final_output))
        print("Correct: " + str(correct) + " Wrong: " + str(wrong) + " Accuracy: " + str(correct/(correct+wrong)))
        average_model_error += correct / (correct + wrong)

    average_model_error /= 10
    print("Average Model Error: " + str(average_model_error))





# Scenario Two of evaluating your ANN. Uses nested Cross Validation.
# Each one of these algorithms are giving 8 training sets and 1 validation set. 
# Cross validation is done for each of them. And each will have an average error 
# for the 9 times they were run. From there the one with the lowest error rate is tested on the test set.
def run_neural_network_compare_models():

    data = k_fold_cross_validation()
     
    data_input = data[0]
    data_labels = data[1]

    # Two different learning rates will be applied to each configuration.
    average_test_error = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    standard_deviation = 0

    # Cross Validation: Concatenating the folds to make one training set.
    for test_fold in range(0, len(data_input)):

        neural_nets = []
        # Removing testfold index from training and validation indexes.
        fold_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #print(str(test_fold))
        fold_indexes.remove(test_fold)
        
        # Error for each configuration. 
        avg_error = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for validation_fold in fold_indexes:
            #print("TEST: " + str(test_fold) + " VAL: " + str(validation_fold))

            combined_folds_input = []
            combined_folds_label = []

            for training_fold in fold_indexes:

                # Removing validation fold index from training fold indexes. 
                if training_fold == validation_fold:
                    continue

                # If training set is empty, set it equal to the first fold. Else concatenate it.
                if len(combined_folds_input) == 0:
                    combined_folds_input = data_input[training_fold]
                    combined_folds_label = data_labels[training_fold]
                else:
                    combined_folds_input = np.concatenate((combined_folds_input, data_input[training_fold]), axis = 0)
                    combined_folds_label = np.concatenate((combined_folds_label, data_labels[training_fold]), axis = 0)

            # Run NNs with different configurations and test them on the validation data.
            # Each configuration is run 9 times and whic)hever has the lowest average epoch error is ran on the test data.
            neural_nets.append(NeuralNetwork(7, [9], 3))
            neural_nets.append(NeuralNetwork(7, [9], 3))
            neural_nets.append(NeuralNetwork(7, [10], 3))
            neural_nets.append(NeuralNetwork(7, [10], 3))
            neural_nets.append(NeuralNetwork(7, [15], 3))
            neural_nets.append(NeuralNetwork(7, [15], 3))
            neural_nets.append(NeuralNetwork(7, [4], 3))
            neural_nets.append(NeuralNetwork(7, [4], 3))
            neural_nets.append(NeuralNetwork(7, [2], 3))
            neural_nets.append(NeuralNetwork(7, [3], 3)) 

            # Training.
            neural_nets[0].train(combined_folds_input, combined_folds_label, 0.02, 0, 0) # Last index is for validation. 
            neural_nets[1].train(combined_folds_input, combined_folds_label, 0.02, 0, 0) # Currently 0, change later.
            neural_nets[2].train(combined_folds_input, combined_folds_label, 0.01, 0, 0)
            neural_nets[3].train(combined_folds_input, combined_folds_label, 0.02, 0, 0)
            neural_nets[4].train(combined_folds_input, combined_folds_label, 0.01, 0, 0)
            neural_nets[5].train(combined_folds_input, combined_folds_label, 0.02, 0, 0)
            neural_nets[6].train(combined_folds_input, combined_folds_label, 0.01, 0, 0)
            neural_nets[7].train(combined_folds_input, combined_folds_label, 0.02, 0, 0)
            neural_nets[8].train(combined_folds_input, combined_folds_label, 0.01, 0, 0)
            neural_nets[9].train(combined_folds_input, combined_folds_label, 0.02, 0, 0)

            # Testing with validation fold.
            avg_error[0] += neural_nets[0].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[1] += neural_nets[1].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[2] += neural_nets[2].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[3] += neural_nets[3].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[4] += neural_nets[4].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[5] += neural_nets[5].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[6] += neural_nets[6].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[7] += neural_nets[7].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[8] += neural_nets[8].validation(data_input[validation_fold], data_labels[validation_fold])
            avg_error[9] += neural_nets[9].validation(data_input[validation_fold], data_labels[validation_fold])
            #print("CUMULATIVE VALIDATION ERRORS: " + str(avg_error))


        # Getting average error from all 9 runs and per sample
        for error in avg_error:
            error /= 9

        # Finding NN with lowest average error.
        lowest_error_index = 0
        for index in range(1, len(avg_error)):
            if avg_error[index] < avg_error[lowest_error_index]:
                lowest_error_index = index

        print("Lowest Error: " + str(lowest_error_index))

        # Run the test fold with the NN with lowest average error.
        correct = 0
        wrong = 0

        for test_case in range(0, len(data_input[test_fold])):
            temp_output = neural_nets[lowest_error_index].forward_prop(data_input[test_fold][test_case])

            if temp_output[0] > temp_output[1] and temp_output[0] > temp_output[2]and temp_output[0] > temp_output[3]and temp_output[0] > temp_output[4]and temp_output[0] > temp_output[5]and temp_output[0] > temp_output[6]: # Healthy
                final_output = 0
            elif temp_output[1] > temp_output[0] and temp_output[1] > temp_output[2]and temp_output[1] > temp_output[3]and temp_output[1] > temp_output[4]and temp_output[1] > temp_output[5]and temp_output[1] > temp_output[6]:
                final_output = 1
            elif temp_output[2] > temp_output[0] and temp_output[2] > temp_output[1]and temp_output[2] > temp_output[3]and temp_output[2] > temp_output[4]and temp_output[2] > temp_output[5]and temp_output[2] > temp_output[6]:
                final_output = 2
            elif temp_output[3] > temp_output[0] and temp_output[3] > temp_output[1]and temp_output[3] > temp_output[2]and temp_output[3] > temp_output[4]and temp_output[3] > temp_output[5]and temp_output[3] > temp_output[6]:
                final_output = 3
            elif temp_output[4] > temp_output[0] and temp_output[4] > temp_output[1]and temp_output[4] > temp_output[2]and temp_output[4] > temp_output[3]and temp_output[4] > temp_output[5]and temp_output[4] > temp_output[6]:
                final_output = 4
            elif temp_output[5] > temp_output[0] and temp_output[5] > temp_output[1]and temp_output[5] > temp_output[2]and temp_output[5] > temp_output[3]and temp_output[5] > temp_output[4]and temp_output[5] > temp_output[6]: # Neurogenic
                final_output = 5
            elif temp_output[6] > temp_output[0] and temp_output[6] > temp_output[1]and temp_output[6] > temp_output[2]and temp_output[6] > temp_output[3]and temp_output[6] > temp_output[4]and temp_output[6] > temp_output[5]: # Neurogenic
                final_output = 6
            else:
                print("equal outputs!")

            if (data_labels[test_fold][test_case][0] == 1 and final_output == 0) or \
               (data_labels[test_fold][test_case][1] == 1 and final_output == 1) or \
               (data_labels[test_fold][test_case][2] == 1 and final_output == 2) or \
               (data_labels[test_fold][test_case][3] == 1 and final_output == 3) or \
               (data_labels[test_fold][test_case][4] == 1 and final_output == 4) or \
               (data_labels[test_fold][test_case][5] == 1 and final_output == 5) or \
               (data_labels[test_fold][test_case][6] == 1 and final_output == 6):
                correct += 1
            else:
                wrong += 1

            #print("TestFold: " + str(test_fold) + " Desired: " + str(data_labels[test_fold][test_case]) + " Result: " + str(final_output))
        print("Correct: " + str(correct) + " Wrong: " + str(wrong) + " Accuracy: " + str(correct/(correct+wrong)))
        average_test_error[test_fold] += correct / (correct + wrong)


    print("Average Test Error: " + str(np.mean(average_test_error)) + " STD: " + str(np.std(average_test_error)))        



def k_fold_cross_validation():

    # Getting data.
    data = read_training_data()

    # Each data fold will have 15 cases for IRIS and 80 cases for EMG equaling 10 folds.
    fold_size = 38
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

    combined_cases = "../AANEM-data/combined-data/healthy&diseased.csv"
    iris = "../Iris Testing Data/bezdekIris-data.csv"
    abalone = "../AANEM-data/combined-data/abalone-data.csv"

    data_shuffle = []

    # Opening data file.
    with open(combined_cases, "r") as file_ptr:
        
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
    
    # Shuffling data around to introduce some randomness.
    random.shuffle(data_shuffle)

    data_input = []
    data_labels = []
    data_muscle_id = []

    # Breaking data into input and label and muscle ID.
    for row in data_shuffle:
        temp_input = []
        temp_label = []

        for i in range(0, len(row) - 2):
        #for i in range(0, len(row) - 1):
            temp_input.append(float(row[i]))
        
        #temp_label.append(float(row[-1])) 
        temp_label.append(float(row[-2])) 

        data_input.append(temp_input)
        data_labels.append(temp_label)
        data_muscle_id.append(float(row[-1]))
        #print("in: " + str(temp_input) + " label: " + str(temp_label) + "muscle num: " + str(float(row[-1])))

    # Preprocessing and converting to numpy arrays.
    preprocessed_data = preprocess_data_0_1(data_input)
    
    for i in range(0, len(data_labels)):
        if data_labels[i] == [0.0]:
            data_labels[i] = [1, 0, 0, 0, 0, 0, 0]
        elif data_labels[i] == [1.0] :
            data_labels[i] = [0, 1, 0, 0, 0, 0, 0]
        elif data_labels[i] == [2.0]:
            data_labels[i] = [0, 0, 1, 0, 0, 0, 0]
        elif data_labels[i] == [3.0]:
            data_labels[i] = [0, 0, 0, 1, 0, 0, 0]
        elif data_labels[i] == [4.0]:
            data_labels[i] = [0, 0, 0, 0, 1, 0, 0]
        elif data_labels[i] == [5.0]:
            data_labels[i] = [0, 0, 0, 0, 0, 1, 0]
        elif data_labels[i] == [6.0]:
            data_labels[i] = [0, 0, 0, 0, 0, 0, 1]
        else:
            print("Data with Incorrect Label: " + str(data_labels[i]))
    

    '''
    # Output for EMG three outputs: Zero means healthy, One means Myogenic, Two means Neurogenic.
    for i in range(0, len(data_labels)):
        if data_labels[i] == [0.0]:
            data_labels[i] = [1, 0, 0]
        elif data_labels[i] == [1.0] or data_labels[i] == [2.0] or data_labels[i] == [3.0]:
            data_labels[i] = [0, 1, 0]
        elif data_labels[i] == [4.0] or data_labels[i] == [5.0] or data_labels[i] == [6.0]:
            data_labels[i] = [0, 0, 1]
        else:
            print("Data with Incorrect Label: " + str(data_labels[i]))
    '''
    
    
    '''
    # Desired output for IRIS data.
    for i in range(0, len(data_labels)):
        if data_labels[i] == [0.0]:
            data_labels[i] = [1, 0, 0]
        elif data_labels[i] == [0.5]:
            data_labels[i] = [0, 1, 0]
        elif data_labels[i] == [1.0]:
            data_labels[i] = [0, 0, 1]
        else:
            print("Data with Incorrect Label: " + str(data_labels[i]))
    '''
    
    '''
    # For Abalone data. 0->Male, 1->Female, 2->Infant
    for i in range(0, len(data_labels)):
        if data_labels[i] == [0.0]:
            data_labels[i] = [1, 0, 0]
        elif data_labels[i] == [1.0]:
            data_labels[i] = [0, 1, 0]
        elif data_labels[i] == [2.0]:
            data_labels[i] = [0, 0, 1]
        else:
            print("Data with Incorrect Label: " + str(data_labels[i]))
    '''


    labels = np.array(data_labels)
    muscle_id = np.array(data_muscle_id)

    return preprocessed_data, labels , muscle_id


# Preprocesses data into 2D numpy array with values between -1:1. Finds the max and min 
# of each variable, then uses it to perform transformation on data.
# x" = ((x - min) / ((max - min) / 2)) - 1.
def preprocess_data_0_1(data):

    # Converting list to a numpy array and getting max and min for each variable(axis 0 gets the columns max value). 
    data = np.array(data)
    mins = data.min(axis = 0)
    maxs = data.max(axis = 0)

    # Transforming data based off min and max.
    for variable in range(0, 7):
        min_val = mins[variable]
        max_val = maxs[variable]

        for row in range(0, len(data)):
            try:
                data[row][variable] = ((data[row][variable] - min_val) / ((max_val - min_val) / 2)) - 1 
            except ZeroDivisionError:
                data[row][variable] = 0

    return data

#np.random.seed(0)
run_neural_network_tune_and_train()



'''
# TASK LIST
#
# - Introduce multiple outputs and check to see that they work correctly.
# - Batch Learning.
# - If the slope of error for epoch stays the same, stop the training (means either its stuck or it isn't improving).
#
'''
































