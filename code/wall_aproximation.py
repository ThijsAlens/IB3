import numpy as np
import time
from Actuator_Interpolation import actuator_interpolation

input_data_path = "snapshot_data"


n_rows_used = 10                        # ex. 10: read row 0, read row 9, read row 19, read row 29
n_columns_used = 20                     # ex. 20: read columns 0-19, read columns 1-20, read columns 2-21
threshold = 1                           # ex. 5: if the average difference differs more then threshold, it is considered a sharp edge 

input_data = np.loadtxt(input_data_path)
sample = np.zeros(n_columns_used)
old_slope= 0
new_slope= 0

def detect_walls():
    for row in range(0, len(input_data), n_rows_used):
        for column in range(len(input_data[row])):
            sample = np.roll(sample, 1)
            sample[0] = input_data[row][column]
            new_slope= np.polyfit(np.array(range(0, n_columns_used, 1)), sample, 1)[0]
            #print(f"new slope= {new_slope}")
            if (new_slope+threshold <= old_slope <= new_slope-threshold):
                print(f"wall found at {column-(n_columns_used/2)}, new_slope = {new_slope}, old_slope = {old_slope}")
            old_slope = new_slope

def depth_estimation(data, size):
    result = []
    for row in range(0, len(data), size):
        row_result = []
        for i in range(0, len(data[row]), size):
            mean = np.mean(data[row:row+size, i:i+size])
            row_result.append(mean)
        result.append(row_result)
    return np.array(result)

def main():
    size = 10           # blocks of size x size are seen as 1
    threshold = 150     # min value of data where it is seen as "close"
    max_value = 255     # max possible value in array
    n_actuators = 6     # number of actuators
    min_voltage = 0
    max_voltage = 5

    mean_estimation = depth_estimation(input_data, size)
    print(f"mean grid of size = {size}x{size} | len = {len(mean_estimation[0])}x{len(mean_estimation)}\n")
    actuator_interpolation(mean_estimation, threshold, max_value, n_actuators, min_voltage, max_voltage)



# time calculation
start_time = time.time()
main()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")