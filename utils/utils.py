import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, Ridge
from mlxtend.feature_selection import SequentialFeatureSelector
from modules.memorycapacity import memorycapacity
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.anova import anova_lm

def find_specific_txt_files(folder_path: str) -> list:
    """
    Scans the given folder for .txt files that contain 'MemoryCapacity' in their names
    but not 'log'.
    
    Parameters:
    - folder_path: str - The path to the folder to be scanned.
    
    Returns:
    - List of paths to the files that match the criteria.
    """
    # List all files in the directory
    all_files = os.listdir(folder_path)
    
    # Compile a regular expression to match the criteria
    pattern = re.compile(r'MemoryCapacity(?!.*log).*\.txt$', re.IGNORECASE)
    
    # Filter files using the regular expression
    matching_files = [file for file in all_files if pattern.search(file)]
    
    # Prepend the folder path to each filename
    full_paths = [os.path.join(folder_path, file) for file in matching_files]
    
    return full_paths

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

def reshape_into_matrix(voltage_differences, electrode_pairs, size):
    """
    Reshapes lists of voltage differences and electrode pairs back into a matrix,
    filling the upper triangular part of the matrix.

    Args:
    voltage_differences (list of numpy arrays): List of arrays containing the voltage differences.
    electrode_pairs (list of tuples): List of tuples containing the electrode pairs.
    size (int): The dimension of the NxN matrix to be filled.

    Returns:
    numpy.ndarray: A 2D array of tuples, where each tuple contains a vector of voltage
                   differences and a tuple of electrode indices, filling the upper
                   triangular part of the matrix.
    """
    # Initialize an empty matrix of tuples
    matrix = np.empty((size, size), dtype=object)

    # Fill the matrix with empty data where no values will be placed (optional)
    for i in range(size):
        for j in range(size):
            matrix[i, j] = (np.array([]), (i, j))

    # Populate the upper triangular matrix
    for diff, pair in zip(voltage_differences, electrode_pairs):
        i, j = pair
        matrix[i, j] = (diff, (i, j))

    return matrix

def compute_voltage_differences(voltage_matrix):
    """
    Computes a matrix where each cell contains a tuple with the vector of voltage 
    differences between electrode pairs and the tuple of their indices.

    Args:
    voltage_matrix (numpy.ndarray): A 2D array where each column represents an electrode
                                    and each row represents a timestep.

    Returns:
    numpy.ndarray: A 2D array where each element (i, j) is a tuple. The first element
                   of the tuple is the vector of voltage differences between electrode i 
                   and electrode j across timesteps, and the second element is the tuple 
                   (i, j) indicating the electrode indices.
    """
    # Ensure the input is a NumPy array
    voltage_matrix = np.array(voltage_matrix)
    
    # Number of electrodes
    num_electrodes = voltage_matrix.shape[1]
    
    # Initialize an empty array to store the tuples
    # Note: We need an object array to store tuples
    voltage_differences = np.empty((num_electrodes, num_electrodes), dtype=object)

    # Compute the differences and store them with electrode indices
    for i in range(num_electrodes):
        for j in range(num_electrodes):
            diff_vector = voltage_matrix[:, i] - voltage_matrix[:, j]
            voltage_differences[i, j] = (diff_vector, (i, j))

    return voltage_differences


def extract_upper_triangular_data(matrix):
    """
    Extracts voltage difference vectors and electrode pairs from the upper triangular
    part of the matrix (excluding the diagonal).

    Args:
    matrix (numpy.ndarray): A 2D array of tuples, where each tuple contains a vector
                            of voltage differences and a tuple of electrode indices.

    Returns:
    tuple: (list of numpy arrays, list of tuples)
           First element is a list of arrays containing the voltage differences.
           Second element is a list of tuples containing the electrode pairs.
    """
    # Number of electrodes (assuming a square matrix)
    size = matrix.shape[0]

    # Lists to store the results
    voltage_differences = []
    electrode_pairs = []

    # Traverse only the upper triangular part, excluding the diagonal
    for i in range(size):
        for j in range(i + 1, size):
            data = matrix[i, j]
            voltage_differences.append(data[0])  # Append the voltage difference vector
            electrode_pairs.append(data[1])      # Append the tuple of electrode indices

    return voltage_differences, electrode_pairs


def sequential_regression_evaluate(states_train: np.array, states_test: np.array, target_train:np.array):
    linear_regressor = LinearRegression()
    
    # If no scoring is passed, default is r2 for regressions
    sfs = SequentialFeatureSelector(linear_regressor,
                                    k_features="best",
                                    forward=True,
                                    floating=False,
                                    cv=5)
    sfs = sfs.fit(states_train, target_train)
    selected_features = list(sfs.k_feature_idx_)
    print(f"Selected features indices: {selected_features}")
    X_train_selected = sfs.transform(states_train)
    X_test_selected = sfs.transform(states_test)

    linear_regressor.fit(X_train_selected, target_train)
    return linear_regressor.predict(X_test_selected)

def ftest_evaluate(states_train: np.array, states_test: np.array, target_train:np.array):
    model = LinearRegression().fit(states_train, target_train)

    # Use statsmodels for F-test
    results = {}
    for i in range(0, states_train.shape[1]):  # Start at 1 to skip the intercept
        results[f'Feature {i}'] = test_feature_relevance(states_train, target_train, i)
    return results


def test_feature_relevance(X, y, feature_index):
    # Model with all features except the one being tested
    baseline_features = np.delete(X, feature_index, axis=1)
    baseline_model = OLS(y, baseline_features).fit()

    # Full model with the feature being tested
    full_model = OLS(y, X).fit()

    # F-test to compare models
    anova_results = anova_lm(baseline_model, full_model)
    return anova_results.iloc[1] 

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

def reformat_measurement_header(s:str) -> str:
    # Check if the string starts with a single digit
    if len(s) > 0 and s[0].isdigit() and (len(s) == 1 or not s[1].isdigit()):
        # Prefix the string with '0' if it starts with a single digit
        return '0' + s
    else:
        # Return the original string if it doesn't start with a single digit
        return s
    
def experiment_from_filename(filename:str) -> str:
    if "memorycapacity" in filename.lower():
        return "memorycapacity"
    elif "tomography" in filename.lower():
        return "tomography"
    elif "custom wave" in filename.lower():
        return "custom wave measurement"
    elif "conductivitymatrix" in filename.lower():
        return "conductivitymatrix"
    else:
        return "undefined"