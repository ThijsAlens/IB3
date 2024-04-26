import serial
import numpy as np
import time
from Actuator_Interpolation import actuator_interpolation
from Find_Passage import Find_Passage

#ser = serial.Serial('/dev/ttyACM0', 9600)

def detect_movement(data, differencial, size, n_rows=-1):
    """
    detect_movement aproximates if something is moving in front of the data.

    Args:
        data : The frame that needs to be processed.
        n_rows | default=-1 : How many rows need to be taken into account when processing. n_rows at the bottom of the frame are used. -1 is for all rows.
        differencial : How much difference must there be between the mean value of the bottom and the motion detected
        size : How big an area needs to be, to be considered as movement.
        
    Returns:
        Boolean : True if movement is detected, False if not.
    """

    if (n_rows == -1):
        n_rows = len(data)
    
    starting_index = len(data)-n_rows
    mean_sum = 0
    for row in range(starting_index, len(data), 1):
        mean_sum += np.mean(data[row])
    mean = mean_sum/n_rows 
    t_data = np.transpose(data)
    counter = 0
    for column in range(len(t_data)):
        if (np.mean(t_data[column]) > (mean + differencial)):
            counter += 1
        else:
            counter = 0
        if (counter >= size):
            return True
    return False

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

    # input_data_path = "data.txt"
    # input_data = np.loadtxt(input_data_path)

    import numpy as np

def data_processing(input_data, info_for_current_frame):

    # If not active, do no processing
    if (info_for_current_frame[8] == 0):
        return info_for_current_frame

    # Seperate data so reference is not considered in processing
    input_data_above = input_data[0:(len(input_data)-30)]
    input_data_below = input_data[(len(input_data)-30):len(input_data)]

    # Initializing the info for the next frame to a default
    """
        array consists of info gathered in this frame, it has the following data:
        [0] = threshold
        [1] = #frames where the surrounding (inside/outside) is consistent
        [2] = inside/outside
        [3] = #frames a passage has been found
        [4] = passage, if found else []
        [5] = data for actuators, if processing is not active []
        [6] = buffer size
        [7] = buffer for frames
        [8] = active (1), not active (0)
        [9] = #frames where movement is detected
    """

    info_for_next_frame = []

    info_for_next_frame.append(0)                            #0
    info_for_next_frame.append(0)                            #1
    info_for_next_frame.append("")                           #2
    info_for_next_frame.append(0)                            #3
    info_for_next_frame.append([])                           #4
    info_for_next_frame.append([])                           #5
    info_for_next_frame.append(5)                            #6
    info_for_next_frame.append(np.zeros((5, 4)).tolist())    #7
    info_for_next_frame.append(1)                            #8
    info_for_next_frame.append(0)                            #9

    # Initializing the variables used to a default value
    size = 10                                           # blocks of size x size are seen as 1
    threshold = 150                                     # min value of data where it is seen as "close"
    passage_threshold = 50                              # max value of data where it is seen as an opening
    min_size_passage = 50                               # min area size of a passage (100 = 10x10)
    inside_outside_threshold = 150                      #when is it considered inside (inside is overall higher => threshold needs to be higher)
    max_value = 255                                     # max possible value in array
    default_data_multiplier = 2                         # default multiplier for the data so it is more accurate after deleting the bottom (reference distance)
    data_multiplier_inside = 2                         # multiplier for the data when it is outside
    movement_threshold = 20                             # how much difference must there be between the mean value of the bottom and the motion detected
    min_size_movement = 10                              # how big an area needs to be, to be considered as movement
    n_actuators = 4                                     # number of actuators
    min_voltage = 0                                     # min voltage of the actuator
    max_voltage = 5                                     # max voltage of the actuator
    buffer_size = 10                                    # size of the buffer
    buffer = info_for_current_frame[7][1:buffer_size]   # init buffer

    inside_outside_modifier = 30                        # when triggered, the threshold is inceased by the modifier

    # Preprocessing: compencate the input data + quantize the frame to be a size x size grid
    input_data_above *= default_data_multiplier

    mean_estimation = quantize_frame(input_data_above, size)
    mean_estimation_bottom = quantize_frame(input_data_below, size)

    # Processing of/interpreting the data
    
    # Surrounding consistent?
    surrounding = detect_surrounding(mean_estimation, inside_outside_threshold)
    info_for_next_frame[2] = surrounding

    if (info_for_current_frame[2] == surrounding):
        info_for_next_frame[1] = info_for_current_frame[1] + 1
    else:
        info_for_next_frame[1] = 0

    # Adjust parameters if consistently inside or outside
    if (info_for_current_frame[1] >= 2):
        if (surrounding == "inside"):
            threshold -= inside_outside_modifier
            mean_estimation *= data_multiplier_inside
        else:
            threshold += inside_outside_modifier
    
    # Movement detection, animation if detected more then 2 times
    movement_flag = False
    movement = detect_movement(mean_estimation_bottom, movement_threshold, min_size_movement)
    if (movement == True and info_for_current_frame[9] >= 2):
        print("Movement detected -> call some animation function")
        movement_flag = True
        # call some animation function
    else:
        info_for_next_frame[9] = 0
    
    # Passage detection (only inside), animation if detected more then 2 times
    passage_flag = False
    if (not movement_flag):
        passage = Find_Passage(mean_estimation, size, passage_threshold, min_size_passage)
        if (passage != [] and info_for_current_frame[2] == "inside"):
            info_for_next_frame[3] = info_for_current_frame[3] + 1
            info_for_next_frame[4] = passage
            if (info_for_current_frame[3] >= 2):
                # a passage must be found twice before it is confirmed as a passage
                print(f"Passage found -> call some animation function\n{passage}")
                passage_flag = True
        else:
            info_for_next_frame[3] = 0
            info_for_next_frame[4] = []
    
    # Default interpolation
    actuator_data = actuator_interpolation(mean_estimation, threshold, max_value, n_actuators, min_voltage, max_voltage)
    buffer.append(actuator_data)
    info_for_next_frame[5] = np.mean(np.array(buffer), axis=0)
    info_for_next_frame[6] = buffer_size
    info_for_next_frame[7] = buffer
    if (not (movement_flag or passage_flag)):
        write_serial(info_for_next_frame[5])
    info_for_next_frame[0] = threshold
    #print(f"mean grid of size = {size}x{size} | len = {len(mean_estimation[0])}x{len(mean_estimation)}\n")

    
    print(f"{info_for_next_frame[2]}, passage: {info_for_next_frame[4]}, data = {info_for_next_frame[5]}, movement: {info_for_next_frame[9]}\n")
    
    return info_for_next_frame

'''
# time calculation
start_time = time.time()
data_processing()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
'''
