import numpy as np
from modules.memorycapacity.memorycapacity import calculate_memory_capacity
from modules.memorycapacity.memorycapacity import calculate_mc_from_df
from utils.utils import read_and_parse_to_df

def test_calculate_memory_capacity() -> None:
    estimated_waveform = np.ones((2,100))
    target_waveform = np.ones((2,100))
    MC, memc = calculate_memory_capacity(estimated_waveform, target_waveform)
    print(MC)
    assert MC ==2

def test_calculate_mc_from_df_linear() -> None:
    path = "tests/test_files/011_INRiMARC_NWN_Pad131M_gridSE_MemoryCapacity_2024_03_29.txt"
    measurement, elec_dict = read_and_parse_to_df(path)
    MC, MC_vec = calculate_mc_from_df(measurement, elec_dict, "linear", 30, "08", "17")
    MC = np.round(MC,1)
    assert MC == 2.3

def test_calculate_mc_from_df_ridge() -> None:
    path = "tests/test_files/011_INRiMARC_NWN_Pad131M_gridSE_MemoryCapacity_2024_03_29.txt"
    measurement, elec_dict = read_and_parse_to_df(path)
    MC, MC_vec = calculate_mc_from_df(measurement, elec_dict, "ridge", 30, "08", "17")
    MC = np.round(MC,1)
    assert MC == 1.9

def test_calculate_mc_from_df_sequential() -> None:
    path = "tests/test_files/011_INRiMARC_NWN_Pad131M_gridSE_MemoryCapacity_2024_03_29.txt"
    measurement, elec_dict = read_and_parse_to_df(path)
    MC, MC_vec = calculate_mc_from_df(measurement, elec_dict, "sequential", 30, "08", "17")
    MC = np.round(MC,1)
    assert MC == 2.2 