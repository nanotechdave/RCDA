import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
    lwav = len(estimated_waveforms)
    print(f"len of estimated waveform is {lwav}")
    MC_values = []
    MemC = 0
    for estimated_waveform, target_waveform in zip(
        estimated_waveforms, target_waveforms
    ):
        print(estimated_waveform)
        # Calculate the covariance and variances
        covariance = np.cov(estimated_waveform, target_waveform)[0, 1]
        print(f"covariance is {covariance}")
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

def plot_forgetting_curve(MC_vec: np.array) -> None:
   
    # Plot the forgetting curve
    plt.plot(range(1, len(MC_vec) + 1), MC_vec)
    plt.xlabel("Delay")
    plt.ylabel("Memory Capacity")
    plt.title("Forgetting Curve")
    return

def read_and_parse_voltages(filename):
    # Read the file into a pandas DataFrame
    df = pd.read_csv(filename, sep=r'\s+')

    # Get voltage column names by filtering out current (I) columns
    voltage_columns = [col for col in df.columns if "V" in col]

    # Create a matrix with the voltage values
    voltage_matrix = df[voltage_columns].values

    return voltage_matrix

def calculate_mc_from_file(path:str, mode: str = "linear") -> float:
    
    #import the data from file 
    OUTPUT_NODES = np.arange(15,16)
    voltage_mat = read_and_parse_voltages(path)
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
                data = voltage_mat[:, 1:n]  
            else:
                target = target[:-k]
                data = voltage_mat[:-k, 1:n]
                
            target = target[10:-10]
            data = data[10:-10, :]

            #split test and train sets
            states_train, states_test, target_train, target_test = train_test_split_time_series(
                data,
                target,
                test_size=0.2,
                )
            
            #compute the prediction
            if mode.lower() == "linear":
                prediction_test = linear_regression_predict(states_train, states_test, target_train)
            elif mode.lower() == "ridge":
                prediction_test = ridge_regression_predict(states_train, states_test, target_train, alpha = 1.0)
            target_vec.append(target_test)
            estimated_vec.append(prediction_test)

        MC, MC_values = calculate_memory_capacity(estimated_vec, target_vec)

        """n_MC.append(MC)
        MC_vec.append(MemC)"""
    return MC
