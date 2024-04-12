import logging

from classes import sample
import pandas as pd

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
    return



if __name__ == "__main__":
    main()
    
