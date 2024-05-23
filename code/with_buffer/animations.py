import numpy as np
import math
import time

import Data_Processing

def startup_function(x):
    """ ease function for the startup"""
    return round(math.sin(math.pi*x), 3)

def create_values(easeFunction, frequency):
    step = 1/(frequency-1)
    values = [easeFunction(i * step) for i in range(frequency)]
    return values

def send_startup_sequence():
    values = create_values(startup_function, 800)
    reverse = values[::-2]

    finalLRValues = list(np.zeros(1400))
    finalMValues = list(np.zeros(1400))

    finalLRValues[:799] = values
    finalLRValues[900:919] = np.ones(20)
    finalLRValues[1000:] = reverse
    
    finalMValues[400:799] = values[:399]
    finalMValues[900:919] = np.ones(20)
    finalMValues[1000:] = reverse

    finalTotal = np.array([finalLRValues, finalMValues, finalMValues, finalLRValues])

    if (not Data_Processing.writingToActuators):
        Data_Processing.writingToActuators = True
        for i in range(1400):
            Data_Processing.write_serial(np.array([finalTotal[0][i], finalTotal[1][i], finalTotal[2][i], finalTotal[3][i]]))
        Data_Processing.writingToActuators = False

def send_passage_sequence(passage):
    final = list(np.zeros(50))
    final = [final[:] for _ in range(4)]
    for i in range(4):
        if ((passage[0][1] >= (64/4)*i) or (passage[1][1] > (64/4)*i+(64/4))):
            final[i][:12] = np.ones(12)*5
            final[i][25:37] = np.ones(12)*5
            
    if (not Data_Processing.writingToActuators):
        Data_Processing.writingToActuators = True
        for i in range(50):
            Data_Processing.write_serial(np.array([final[0][i], final[1][i], final[2][i], final[3][i]]))
        Data_Processing.writingToActuators = False