
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import find_specific_txt_files


def train_test_split_time_series(data: np.array, target: np.array, test_size=0.2):
    """ Splits data and target into training and test set."""
    split_index = int(len(data) * (1 - test_size))
    data_train = data[:split_index, :]
    data_test = data[split_index:, :]
    target_train = target[:split_index]
    target_test = target[split_index:]
    return data_train, data_test, target_train, target_test

def linear_regression_predict(states_train: np.array, states_test:np.array, target_train: np.array) -> np.array:
    """ Performs linear regression using the states to match the target.
     Returns the predicted waveform"""
    states_train = np.array(states_train)
    states_test = np.array(states_test)
    target_train = np.array(target_train)
    lr = LinearRegression()
    lr.fit(states_train, target_train)
    return lr.predict(states_test)

def ridge_regression_predict(states_train: np.array, states_test: np.array, target_train: np.array, alpha: float = 1.0) -> np.array:
    """Performs ridge regression using the states to match the target.
    Returns the predicted waveform."""
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(states_train, target_train)
    return ridge_reg.predict(states_test)

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
        if variance_target == 0 and variance_estimate == 0:
            MC_k = 1
        else:
            MC_k = covariance**2 / (variance_estimate * variance_target)

        MC_values.append(MC_k)
        # Add to the total MC
        MemC += MC_k

    return MemC, MC_values

def active_electrode_analysis(measurements:pd.DataFrame, electrode_status: dict, MC_full: float, kDelay: int = 30) -> np.array:
    voltage_columns = [col for col in measurements.columns if '_V[' in col]
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
                data_train, data_test, target_train, target_test = train_test_split_time_series(data[100:,:], target[100:], 0.2)
                predict_test = linear_regression_predict(data_train, data_test, target_train)
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

def extract_voltage_matrix(df: pd.DataFrame) -> np.array:
    """
    Extracts voltage measurements from the DataFrame and creates a matrix of voltages.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing the measurement data.
    
    Returns:
    - np.array - A 2D numpy array (matrix) containing the voltage measurements.
    """
    # Filter columns that contain '_V[' in their column name, indicating a voltage measurement
    voltage_columns = [col for col in df.columns if '_V[' in col]
    
    # Select only the voltage columns from the DataFrame
    voltage_df = df[voltage_columns]
    
    # Convert the DataFrame to a numpy array (matrix)
    voltage_matrix = voltage_df.to_numpy()
    
    return voltage_matrix

def read_and_parse_to_df(filename: str, bias_electrode: str = '08', gnd_electrode: str = '17'):
    # Read the file into a pandas DataFrame
    assert len(bias_electrode)==2 and len(gnd_electrode) ==2 and int(bias_electrode)>0 and int(bias_electrode)<64, "bias_electrode and gnd_electrode must be 2-digit numbers between 01 and 64"
    df = pd.read_csv(filename, sep=r'\s+')
    for col in df.columns:
        df.rename(columns={col: reformat_measurement_header(col)}, inplace=True)
    
    elec_dict = {}
    elec_dict["bias"] = [reformat_measurement_header(str(bias_electrode))]
    elec_dict["gnd"] = [reformat_measurement_header(str(gnd_electrode))]
    elec_dict["float"] = [col.split("_",1)[0] for col in df.columns if isFloat(col, bias_electrode, gnd_electrode)]
    return df, elec_dict

def isFloat(col:str, bias:str, gnd:str) -> bool:
    return ((bias not in col) and (gnd not in col) and "Time" not in col and "I" not in col)

"""def calculate_mc_from_file(path:str, model: str = "linear") -> float:
    
    #import the data from file 
    OUTPUT_NODES = np.arange(15,16)
    voltage_mat = read_and_parse_voltages_to_mat(path)
    voltage_mat = voltage_mat[10:]
    estimated_vec = []
    target_vec = []
    n_MC = [0, 0, 0]
    MC_vec = []

    #for each total number of output nodes
    for n in OUTPUT_NODES:
        estimated_vec = []
        target_vec = []

        #for each k-delay
        for k in np.arange(1, 30): #construct the target and data matrices
            target = np.roll(voltage_mat[:, 0], k)
            if k == 0:
                target = target[:]
                data = voltage_mat[:, 1:n+1]
                  
            else:
                target = target[:-k]
                data = voltage_mat[:-k, 1:n+1]
                
            target = target[10:-10]
            data = data[10:-10, :]

            #split test and train sets
            states_train, states_test, target_train, target_test = train_test_split_time_series(
                data,
                target,
                test_size=0.2,
                )
            
            #compute the prediction
            if model.lower() == "linear":
                prediction_test = linear_regression_predict(states_train, states_test, target_train)
            elif model.lower() == "ridge":
                prediction_test = ridge_regression_predict(states_train, states_test, target_train, alpha = 1.0)
            target_vec.append(target_test)
            estimated_vec.append(prediction_test)

        MC, MC_values = calculate_memory_capacity(estimated_vec, target_vec)

        n_MC.append(MC)
        MC_vec.append(MemC)
    return MC"""

def folder_analysis_MC(path: str):
    
    filenames = find_specific_txt_files(path)
    MC_vec = []
    for filename in filenames:
        MC_val, _ = calculate_mc_from_file(filename)
        MC_vec.append(MC_val)
    return MC_vec

def calculate_mc_from_file(path:str , model:str = "linear", kdelay:int = 30, bias_elec:str = "08", gnd_elec:str = "17"):
    # fetch data into a measurement dataframe and an electrode role dictionary
    measurement, elec_dict = read_and_parse_to_df(path, bias_elec, gnd_elec)
    bias_voltage = []
    gnd_voltage = []
    float_voltage = []
    target_vec = []
    estimated_vec = []

    # fill matrices for ease of computation
    bias_voltage, gnd_voltage, float_voltage = fillVoltageMatFromDf(measurement, elec_dict)
    for k in range(kdelay):
        if k!=0:
            # construct the delayed waveform
            bias_voltage_del, float_voltage_del, gnd_voltage_del = delayWaveforms(bias_voltage, float_voltage, gnd_voltage, k)
            #divide test and train set
            states_train, states_test, target_train, target_test = train_test_split_time_series(
                        float_voltage_del,
                        bias_voltage_del,
                        test_size=0.2,
                        )
            # train model
            if model.lower() == "linear":
                prediction_test = linear_regression_predict(states_train, states_test, target_train)
            elif model.lower() == "ridge":
                prediction_test = ridge_regression_predict(states_train, states_test, target_train, alpha = 1.0)

            # arrays must be flattened from [[1],[2],[3]] to [1,2,3]
            prediction_test = np.array(prediction_test).flatten()
            target_test = np.array(target_test).flatten()
            target_vec.append(target_test)
            estimated_vec.append(prediction_test)
            # return MC and MC for each k delay
    return calculate_memory_capacity(estimated_vec, target_vec)

def fillVoltageMatFromDf(measurement:pd.DataFrame, elec_dict:dict) -> list[np.array]:
    bias_voltage = []
    gnd_voltage = []
    float_voltage = []
    
    for col in measurement.columns:
        if any((str(elec) in col and "V" in col) for elec in elec_dict["float"]):
            float_voltage.append(measurement[col].values)
        elif any((str(elec) in col and "V" in col) for elec in elec_dict["bias"]):
            bias_voltage.append(measurement[col].values)
        elif any((str(elec) in col and "V" in col) for elec in elec_dict["gnd"]):
            gnd_voltage.append(measurement[col].values)

    return np.array(bias_voltage).T, np.array(gnd_voltage).T, np.array(float_voltage).T

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

def reformat_measurement_header(s:str) -> str:
    # Check if the string starts with a single digit
    if len(s) > 0 and s[0].isdigit() and (len(s) == 1 or not s[1].isdigit()):
        # Prefix the string with '0' if it starts with a single digit
        return '0' + s
    else:
        # Return the original string if it doesn't start with a single digit
        return s
"""
path = "/Users/davidepilati/Library/CloudStorage/OneDrive-PolitecnicodiTorino/PhD/Misure/InrimARC/NWN_Pad130M/"
filename = "011_INRiMARC_NWN_Pad130M_gridSE_MemoryCapacity_2024_03_28.txt"
filepath = path+filename

MC, MCval = calculate_mc_from_file(filepath, "linear", 30)
MC_val = folder_analysis_MC(path)

measurement, electrode_status = read_and_parse_to_df(filepath)
active_electrode_analysis(measurement, electrode_status, calculate_mc_from_file(filepath), 30)
print(calculate_mc_from_file(filepath))
print(read_and_parse_to_df(filepath)[1])"""
#print(folder_analysis_MC(path))