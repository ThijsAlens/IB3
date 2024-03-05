import numpy as np
import time
from Actuator_Interpolation import actuator_interpolation

input_data_path = "snapshot_data"
input_data = np.loadtxt(input_data_path)

def detect_movement(data, differencial, n_rows=10):
    """
    detect_movement aproximates if something is moving in front of the data.

    Args:
        data : The frame that needs to be processed.
        n_rows | default=10 : How many rows need to be taken into account when processing. n_rows at the bottom of the frame are used.
        differencial : How much difference must there be between the mean value of the bottom and the motion detected
    
    Returns:
        npArray : with 0 and 1 where a 1 is movement detected and 0, no movement
    """

    starting_index = len(data)-n_rows
    mean_sum = 0
    for row in range(starting_index, len(data), 1):
        mean_sum += np.mean(data[row])
    mean = mean_sum/n_rows 
    t_data = np.transpose(data)
    r_data = np.empty(len(data))
    for column in range(len(t_data)):
        if (np.mean(t_data[column]) > (mean + differencial)):
            r_data[column] = 1
        else:
            r_data[column] = 0
    return r_data

def detect_surrounding(data, inside_outside_threshold, n_rows=10):
    """
    detect_surrounding aproximates the surrounding where the image is taken.

    Args:
        data : The frame that needs to be processed.
        n_rows | default=10 : How many rows need to be taken into account when processing. n_rows at the top of the frame are used.
        inside_outside_threshold : when is it seen as outside/inside, the higher the value the more likely it is to return outside.

    Returns:
        String : "inside" / "outside".
    """

    inside = 0
    outside = 0
    # open_space = 0    extra parameter to identify?
    for row in range(0, n_rows):
        if (np.mean(data[row]) >= inside_outside_threshold):    # >= threshold == inside
            inside += 1
        elif (np.mean(data[row]) < inside_outside_threshold):
            outside += 1
    if (inside > outside):
        return "inside"
    elif (inside < outside):
        return "outside"
    
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

def quantize_frame(data, size):
    """
    quantize_frame quantizes the frame into chunks of size x size.
    ex : data is 400x200, size is 10, result is 40x20 where every original 10x10 now is a 1x1 (mean of 10x10).

    Args:
        data : The frame that needs to be processed.
        size : How big are the squares that need to be processed as 1.
        
    Returns:
        npArray : original frame quantized according to the parameter size.
    """

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

    mean_estimation = quantize_frame(input_data, size)
    print(f"mean grid of size = {size}x{size} | len = {len(mean_estimation[0])}x{len(mean_estimation)}\n")
    actuator_interpolation(mean_estimation, threshold, max_value, n_actuators, min_voltage, max_voltage)


# time calculation
start_time = time.time()
main()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
