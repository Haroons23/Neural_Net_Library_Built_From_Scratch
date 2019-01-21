import numpy as np

# Reading in data from CSV file and storing it into a 2D array.
def read_training_data():

    healthy_cases = "AANEM-data/combined-data/healthy.csv"
    diseased_cases = "AANEM-data/combined-data/diseased.csv"
    data = []

    # Opening data file.
    with open(healthy_cases, "r") as file_ptr:
        
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
                for i in line:
                    single_unit_data.append(float(i))
                data.append(single_unit_data)

    preprocessed_data = preprocess_data_0_1(data)

    return 


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
    


read_training_data()










