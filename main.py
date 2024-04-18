import logging

from classes import sample
import pandas as pd
import numpy as np
from utils import utils as ut
from modules.memorycapacity import memorycapacity

PATH = "/Users/davidepilati/Library/CloudStorage/OneDrive-PolitecnicodiTorino/PhD/Misure/InrimARC/NWN_Pad130M/"
SAMPLE_NAME = "130M" 


def main():
    """ logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        filename = "basic.log",
    )
    logging.debug("this is a debug message")
    logging.info("now we are printing info")
    logging.warning("this is a warning")
    logging.error("this is an error")
    logging.critical("critical message") """

    NWN_130M = sample.Sample(SAMPLE_NAME, PATH)
    for meas in NWN_130M.measurements:
        print(meas.number, meas.experiment)
        if meas.experiment=="memorycapacity":
            print(f"scores: {meas.elec_scores}")
    """ path = "tests/test_files/011_INRiMARC_NWN_Pad131M_gridSE_MemoryCapacity_2024_03_29.txt"
    measurement, elec_dict = ut.read_and_parse_to_df(path)
    MC, MC_vec = memorycapacity.calculate_mc_from_df(measurement, elec_dict, "ftest", 30, "08", "17")
    MC = np.round(MC,1) """
    
    return



if __name__ == "__main__":
    main()
    
