import serial
import numpy as np
import time
from Actuator_Interpolation import actuator_interpolation
from Find_Passage import Find_Passage

#ser = serial.Serial('/dev/ttyACM0', 9600)

def init_data_processing():
    """
    initialize the global variables

    writingToActuators : if false, enables functions to write data to the actuators, if true, some function is writing data, so no other function can
    ser : SSerial object udes to write to arduino-file
    """
    global writingToActuators
    writingToActuators = False
    global ser
    #ser = serial.Serial('/dev/ttyACM0', 9600)

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

def detect_surrounding(data, inside_outside_threshold, n_rows=4):
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
"""    
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
"""

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

def write_serial(input_data):
    print("start write\n")
    if (ser.isOpen() == False):
        ser.open()
    
    res = ""
    for i in range(len(input_data)-1):
        res += f"{input_data[i]},"
    res += f"{input_data[len(input_data)-1]}"
    res += "\n"
    ser.write(res.encode())
    print(res)
    #time.sleep(0.1)
    ser.close()
    print("end write\n")

def data_processing(input_data, info_for_current_frame):

    input_data_path = "data.txt"
    # input_data = np.loadtxt(input_data_path)

    input_data_above = input_data[0:(len(input_data)-30)]
    input_data_below = input_data[(len(input_data)-30):len(input_data)]

    info_for_next_frame = list(range(7))
    info_for_next_frame[2] = ""
    """
        array consists of info gathered in this frame, it has the following data:
        [0] = threshold
        [1] = #frames where the surrounding (inside/outside) is consistent
        [2] = inside/outside
        [3] = #frames a passage has been found
        [4] = passage, if found else []
        [5] = data for actuators, if processing is not active []
        [6] = active (1), not active (0)
    """

    size = 10           # blocks of size x size are seen as 1
    threshold = 150     # min value of data where it is seen as "close"
    passage_threshold = 50  # max value of data where it is seen as an opening
    min_size_passage = 50   # min area size of a passage (100 = 10x10)
    inside_outside_threshold = 150  #when is it considered inside (inside is overall higher => threshold needs to be higher)
    max_value = 255     # max possible value in array
    default_data_multiplier = 2    # default multiplier for the data so it is moreaccurate after data croping
    data_multiplier_outside = 2    # multiplier for the data when it is outside
    n_actuators = 4     # number of actuators
    min_voltage = 0
    max_voltage = 5

    #modifiers for the threshold
    inside_outside_modifier = 30    # when triggered, the threshold is inceased by the modifier


    # preprocessing
    input_data_above *= default_data_multiplier

    mean_estimation = quantize_frame(input_data_above, size)

    # processing

    if (info_for_current_frame[6] == 1):
        info_for_next_frame[6] = 1

    surrounding = detect_surrounding(mean_estimation, inside_outside_threshold)
    info_for_next_frame[2] = surrounding
    if (info_for_current_frame[2] == surrounding):
        info_for_next_frame[1] = info_for_current_frame[1] + 1
    else:
        info_for_next_frame[1] = 0
    if (info_for_current_frame[1] >= 2):
        if (surrounding == "inside"):
            threshold -= inside_outside_modifier
        else:
            threshold += inside_outside_modifier
            mean_estimation *= data_multiplier_outside
    
    passage = Find_Passage(mean_estimation, size, passage_threshold, min_size_passage)
    if (passage != [] and info_for_current_frame[2] == "inside"):
        # possibly a function that checks if it is the same passage as the previous, but more sensors needed
        info_for_next_frame[3] = info_for_current_frame[3] + 1
        info_for_next_frame[4] = passage
        if (info_for_current_frame[3] >= 2):
            # a passage must be found twice before it is confirmed as a passage
            info_for_next_frame[5] = passage
            print(f"Passage found -> call some animation function\n{passage}")
    else:
        info_for_next_frame[3] = 0
        if (info_for_current_frame[6] == 1):
            actuator_data = actuator_interpolation(mean_estimation, threshold, max_value, n_actuators, min_voltage, max_voltage)
            info_for_next_frame[5] = actuator_data
            if (not writingToActuators):
                write_serial(actuator_data)
        else:
            info_for_next_frame[5] = []
            print(f"Processing not active")
    info_for_next_frame[0] = threshold
    #print(f"mean grid of size = {size}x{size} | len = {len(mean_estimation[0])}x{len(mean_estimation)}\n")

    
    print(info_for_next_frame)
    
    return info_for_next_frame

'''
# time calculation
start_time = time.time()
data_processing()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
'''
