import numpy as np

def crop_data_for_actuators(data, n_actuators):
    devider = len(data[0]) // (n_actuators-1)  # int conversion could be a problem later # n-1: devide in 5 so 4 divisions
    result = []

    for row in range(len(data)):
        row_result = []
        for n in range(0, len(data[row]), devider):
            row_result.append(sum(data[row][n:n+devider-1])/len(data[row][n:n+devider-1]))
        result.append(row_result)

    return np.array(result)

def translate_data_to_volatages(data, threshold, max_value, min_voltage, max_voltage):
    t_data = np.transpose(data)
    c_data = np.empty(len(t_data))
    v_data = np.empty(len(t_data))
    for actuator in range(len(t_data)):
        c_data[actuator] = np.mean(t_data[actuator])
        print(f"max = {np.max(c_data[actuator])}")
        if c_data[actuator] < threshold:
            v_data[actuator] = 0
        else:
            print((c_data[actuator] - threshold) / (max_value - threshold))
            v_data[actuator] = ((c_data[actuator] - threshold) / (max_value - threshold)) * (max_voltage - min_voltage) + min_voltage
    return np.transpose(v_data)

def actuator_interpolation(mean_estimation, threshold, max_value, n_actuators, min_voltage, max_voltage):
    print("Actuator_Interpolation")
    croped_data = crop_data_for_actuators(mean_estimation, n_actuators)
    actuator_data = translate_data_to_volatages(croped_data, threshold, max_value, min_voltage, max_voltage)
    print(f"Actuator voltages: {actuator_data}")
