import numpy as np
from modules.memorycapacity import train_test_split_time_series
from modules.memorycapacity import calculate_memory_capacity
from modules.memorycapacity import getMcMeasurement


def test_train_test_split_time_series() -> None:
    data = np.ones((100,16))
    target = np.ones(100)
    data_train, data_test, target_train, target_test = train_test_split_time_series(data, target, 0.3)
    isDataCorrect = data_train.shape == (70,16) and data_test.shape == (30,16)
    isTargetCorrect = target_train.shape == (70,) and target_test.shape == (30,)
    assert isDataCorrect and isTargetCorrect


def test_calculate_memory_capacity() -> None:
    estimated_waveform = np.ones((2,100))
    target_waveform = np.ones((2,100))
    MC, memc = calculate_memory_capacity(estimated_waveform, target_waveform)
    print(MC)
    assert MC ==2

def test_getMCmeasurement() -> None:
    MC = getMcMeasurement("/Users/davidepilati/Library/CloudStorage/OneDrive-PolitecnicodiTorino/PhD/Misure/InrimARC/NWN_Pad131M/011_INRiMARC_NWN_Pad131M_gridSE_MemoryCapacity_2024_03_29.txt")
    MC = np.round(MC,1)
    assert MC == 2.2