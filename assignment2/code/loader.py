import os
import numpy as np

trainfn = "./digitstrain.txt"
validfn = "./digitsvalid.txt"
testfn = "./digitstest.txt"

def loaddata():
    contentlist = []
    list_data = []
    with open (trainfn) as f:
        content = f.readlines()
        for s in content:
            contentlist.append(s[:-1].split(','))
        for l in contentlist:
            data = [float(i) for i in l]
            list_data.append(data[:-1])

    list_data = np.array(list_data)
    training_data = list_data

    contentlist = []
    list_data = []
    list_result = []

    with open (validfn) as f:
        content = f.readlines()
        for s in content:
            contentlist.append(s[:-1].split(','))
        for l in contentlist:
            data = [float(i) for i in l]
            list_data.append(data[:-1])

    list_data = np.array(list_data)
    validation_data = list_data

    contentlist = []
    list_data = []
    list_result = []

    with open (testfn) as f:
        content = f.readlines()
        for s in content:
            contentlist.append(s[:-1].split(','))
        for l in contentlist:
            data = [float(i) for i in l]
            list_data.append(data[:-1])

    list_data = np.array(list_data)
    testing_data = list_data

    return training_data, validation_data, testing_data




    
  
        
    

    
    
    
