import numpy as np

def crop_data_for_actuators(data, n_actuators):
    """
    crop_data_for_actuators quantizes the frame into chunks for n_actuators.
    ex : data is 40x20, n_actuators is 5 -> return an array with 5 values where each value is the mean of an 8x20 part of the original frame.

    Args:
        data : The frame that needs to be processed.
        n_actuators : The ammount of actuators the frame needs to be split between.
        
    Returns:
        npArray : original frame quantized according to the parameter n_actuators.
    """

    devider = len(data[0]) // (n_actuators-1)  # int conversion could be a problem later # n-1: devide in 5 so 4 divisions
    result = []

    for row in range(len(data)):
        row_result = []
        for n in range(0, len(data[row]), devider):
            row_result.append(sum(data[row][n:n+devider-1])/len(data[row][n:n+devider-1]))
        result.append(row_result)

    return np.array(result)

def translate_data_to_volatages(data, threshold, max_value, min_voltage, max_voltage):
    """
    translate_data_to_volatages interpolates the data between a min and max voltage when a min threshold is exceeded.

    Args:
        data : The frame that needs to be processed.
        threshold : The minimum value of data to be seen as "close enough" (everything below will not provoke a respons).
        max_value : The max value that the data can possibly be.
        min_voltage : The minimum voltage of the actuator.
        max_voltage : The maximum voltage of the actuator.
        
    Returns:
        npArray : original frame interpolated to be volatges
    """

    t_data = np.transpose(data)
    c_data = np.empty(len(t_data))
    v_data = np.empty(len(t_data))
    for actuator in range(len(t_data)):
        c_data[actuator] = np.mean(t_data[actuator])
        #print(f"max = {np.max(c_data[actuator])}")
        if c_data[actuator] < threshold:
            v_data[actuator] = 0
        else:
            #print((c_data[actuator] - threshold) / (max_value - threshold))
            v_data[actuator] = ((c_data[actuator] - threshold) / (max_value - threshold)) * (max_voltage - min_voltage) + min_voltage
    return np.transpose(v_data)

def actuator_interpolation(mean_estimation, threshold, max_value, n_actuators, min_voltage, max_voltage):
    #print("Actuator_Interpolation")
    croped_data = crop_data_for_actuators(mean_estimation, n_actuators)
    actuator_data = translate_data_to_volatages(croped_data, threshold, max_value, min_voltage, max_voltage)
    print(f"Actuator voltages: {actuator_data}")
