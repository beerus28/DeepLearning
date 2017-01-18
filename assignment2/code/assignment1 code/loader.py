import os
import numpy as np

trainfn = "../digitstrain.txt"
validfn = "../digitsvalid.txt"
testfn = "../digitstest.txt"

def loaddata():
    contentlist = []
    list_data = []
    list_result = []
    with open (trainfn) as f:
        content = f.readlines()
        for s in content:
            contentlist.append(s[:-1].split(','))
        for l in contentlist:
            data = [[float(i)] for i in l]
            list_data.append(data[:-1])
            result = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
            result[int(data[-1][0])][0] = 1
            list_result.append(result)
    list_data = np.array(list_data)
    list_result= np.array(list_result)
    training_data = zip(list_data, list_result)

    contentlist = []
    list_data = []
    list_result = []

    with open (validfn) as f:
        content = f.readlines()
        for s in content:
            contentlist.append(s[:-1].split(','))
        for l in contentlist:
            data = [[float(i)] for i in l]
            list_data.append(data[:-1])
            result = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
            result[int(data[-1][0])][0] = 1
            list_result.append(result)
    list_data = np.array(list_data)
    list_result= np.array(list_result)
    validation_data = zip(list_data, list_result)

    contentlist = []
    list_data = []
    list_result = []

    with open (testfn) as f:
        content = f.readlines()
        for s in content:
            contentlist.append(s[:-1].split(','))
        for l in contentlist:
            data = [[float(i)] for i in l]
            list_data.append(data[:-1])
            result = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
            result[int(data[-1][0])][0] = 1
            list_result.append(result)
    list_data = np.array(list_data)
    list_result= np.array(list_result)
    testing_data = zip(list_data, list_result)

    return training_data, validation_data, testing_data




    
  
        
    

    
    
    
