
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.utils as ut
import logging

logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        filename = "basic.log",
    )

def calculate_memory_capacity(estimated_waveforms: list[np.array], target_waveforms: list[np.array]) -> float:
    """
    Calculate the memory capacity of a system given the estimated and target waveforms for each delay.

    Parameters:
    estimated_waveforms (list of np.array): The estimated waveforms from the system for each delay.
    target_waveforms (list of np.array): The target waveforms for each delay.

    Returns:
    float: The memory capacity of the system.
    """
    assert len(estimated_waveforms) == len(
        target_waveforms
    ), "Input waveforms must be the same length"
    MC_values = []
    MemC = 0
    for estimated_waveform, target_waveform in zip(
        estimated_waveforms, target_waveforms
    ):
        # Calculate the covariance and variances
        covariance = np.cov(estimated_waveform, target_waveform)[0, 1]
        variance_estimate = np.var(estimated_waveform)
        variance_target = np.var(target_waveform)

        # Calculate the MC for this delay
        if variance_target == 0 or variance_estimate == 0:
            MC_k = 1
        else:
            MC_k = covariance**2 / (variance_estimate * variance_target)

        MC_values.append(MC_k)
        # Add to the total MC
        MemC += MC_k
    
    return MemC, MC_values



def active_electrode_analysis(measurements:pd.DataFrame, electrode_status: dict, MC_full: float, kDelay: int = 30) -> np.array:
    voltage_columns = [col for col in measurements.columns if "_V[" in col]
    mc_without_electrode = pd.DataFrame()
    electrode_influence = pd.DataFrame()
    for idx, col in enumerate(voltage_columns):
        if str(electrode_status["bias"]) in col or str(electrode_status["gnd"]) in col:
            pass
        else:
            temp_v = voltage_columns.copy()
            removed_col = col.split("_",1)[0]
            temp_v.pop(idx)
            newdf = measurements.drop(columns=[col])
            bias_column = newdf[temp_v[0]]
            nodes_columns = newdf[temp_v[1:len(temp_v)-1]]
            bias_voltages = bias_column.to_numpy()
            nodes_voltage = []
            target_test_array = []
            predict_test_array = []
            for col_node in nodes_columns:
                nodes_voltage.append(nodes_columns[col_node].to_numpy())
            nodes_voltage = np.array(nodes_voltage).T
            #for each k-delay
            for k in np.arange(1, kDelay): #construct the target and data matrices
                target = np.roll(bias_voltages, k)
                if k == 0:
                    target = target[:]
                    data = nodes_voltage[:, 1:13]  
                else:
                    target = target[:-k]
                    data = nodes_voltage[:-k, 1:13]
                    
                target = target[10:-10]
                data = data[10:-10, :]
                data_train, data_test, target_train, target_test = ut.train_test_split_time_series(data[100:,:], target[100:], 0.2)
                predict_test = ut.linear_regression_predict(data_train, data_test, target_train)
                target_test_array.append(target_test)
                predict_test_array.append(predict_test)
            MC, MC_vec = calculate_memory_capacity(predict_test_array, target_test_array)
            mc_without_electrode[col] = [MC]
            electrode_influence[col] = [MC_full - MC]
    print(mc_without_electrode.head())
    print(electrode_influence.head())

def plot_forgetting_curve(MC_vec: np.array) -> None:
   
    # Plot the forgetting curve
    plt.plot(range(1, len(MC_vec) + 1), MC_vec)
    plt.xlabel("Delay")
    plt.ylabel("Memory Capacity")
    plt.title("Forgetting Curve")
    plt.show()
    return

def folder_analysis_MC(path: str):
    
    filenames = ut.find_specific_txt_files(path)
    MC_vec = []
    for filename in filenames:
        data, elec = ut.read_and_parse_to_df(filename)
        MC_val, _ = calculate_mc_from_df(data, elec)
        MC_vec.append(MC_val)
    return MC_vec

def ftest_from_matrices(input: np.array, voltages: list[np.array], model:str = "linear", kdelay:int = 2):
    bias_voltage = input
    float_voltage = voltages
    target_vec = []
    estimated_vec = []
    gnd_voltage = input # fake, only for delay compatibility
    elec_scores = np.zeros(len(float_voltage[0]))
    print(f"len voltages: {len(voltages)}")
    print(f"len voltages[0]: {len(voltages[0])}")
    for k in range(kdelay):
        print(f"k: {k}")
        if k!=0:
            # construct the delayed waveform
            bias_voltage_del, float_voltage_del, gnd_voltage_del = delayWaveforms(bias_voltage, float_voltage, gnd_voltage, k)
            #divide test and train set
            states_train, states_test, target_train, target_test = ut.train_test_split_time_series(
                        float_voltage_del,
                        bias_voltage_del,
                        test_size=0.2,
                        )
            # train model
            features_score = ut.ftest_evaluate(states_train, states_test, target_train)
            
            for idx, key in enumerate(features_score.keys()):
                score_series = features_score[key]
                elec_scores[idx] += score_series["F"]  
    return elec_scores



def calculate_mc_from_df(measurement:pd.DataFrame , elec_dict:dict, model:str = "linear", kdelay:int = 30, bias_elec:str = "08", gnd_elec:str = "17"):
    # fetch data into a measurement dataframe and an electrode role dictionary
    bias_voltage = []
    gnd_voltage = []
    float_voltage = []
    target_vec = []
    estimated_vec = []

    # fill matrices for ease of computation
    bias_voltage, gnd_voltage, float_voltage = ut.fillVoltageMatFromDf(measurement, elec_dict)
    for k in range(kdelay):
        if k!=0:
            # construct the delayed waveform
            bias_voltage_del, float_voltage_del, gnd_voltage_del = delayWaveforms(bias_voltage, float_voltage, gnd_voltage, k)
            #divide test and train set
            states_train, states_test, target_train, target_test = ut.train_test_split_time_series(
                        float_voltage_del,
                        bias_voltage_del,
                        test_size=0.2,
                        )
            # train model
            if model.lower() == "linear":
                prediction_test = ut.linear_regression_predict(states_train, states_test, target_train)
            elif model.lower() == "sequential":
                prediction_test = ut.sequential_regression_evaluate(states_train, states_test, target_train)
            elif model.lower() == "ridge":
                prediction_test = ut.ridge_regression_predict(states_train, states_test, target_train, alpha = 1.0)
            """ if True == True:
                print(f"states shape: {states_train.shape[1]}")
                features_score = ut.ftest_evaluate(states_train, states_test, target_train) 
                for key in features_score.keys():
                    series = features_score[key]
                    print(f"{key}: {series['F']}") """


            # arrays must be flattened from [[1],[2],[3]] to [1,2,3]
            prediction_test = np.array(prediction_test).flatten()
            target_test = np.array(target_test).flatten()
            target_vec.append(target_test)
            estimated_vec.append(prediction_test)
            # return MC and MC for each k delay
    return calculate_memory_capacity(estimated_vec, target_vec)

def delayWaveforms(bias_voltage:np.array, float_voltage:list[np.array] , gnd_voltage:np.array, kdelay:int) -> np.array:
    bias_voltage = np.roll(bias_voltage[:], kdelay)
    
    float_voltage = np.array(float_voltage)
    if kdelay == 0:
        bias_voltage = bias_voltage[:]
            
    else:
        bias_voltage = bias_voltage[:-kdelay]
        float_voltage = float_voltage[:-kdelay, :]
        gnd_voltage = gnd_voltage[:-kdelay]
    
    bias_voltage = bias_voltage[10:-10]
    
    float_voltage = float_voltage[10:-10, :]
    gnd_voltage = gnd_voltage[10:-10]
    return bias_voltage, float_voltage, gnd_voltage



""" path = "/Users/davidepilati/Library/CloudStorage/OneDrive-PolitecnicodiTorino/PhD/Misure/InrimARC/NWN_Pad130M/"
filename = "011_INRiMARC_NWN_Pad130M_gridSE_MemoryCapacity_2024_03_28.txt"
filepath = path+filename

MC, MCval = calculate_mc_from_file(filepath, "linear", 30)
MC_val = folder_analysis_MC(path)

measurement, electrode_status = read_and_parse_to_df(filepath)
active_electrode_analysis(measurement, electrode_status, calculate_mc_from_file(filepath), 30)
print(calculate_mc_from_file(filepath))
print(read_and_parse_to_df(filepath)[1])"""
#print(folder_analysis_MC(path)) """



