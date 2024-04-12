from modules.memorycapacity import memorycapacity
from classes.measurement import Measurement
import os
from utils import utils as ut
import re

class Sample():
    """ Sample data class. Instance is an object containing all the measurements of a sample. """
    def __init__(self, name:str, path:str):
        self.name = name
        self.MC_vec = memorycapacity.folder_analysis_MC(path)
        self.measurements = self.fetch_folder(path)
        return
    
    def __str__(self):
        objstr = f"Sample NWN_Pad{self.name}"
        return objstr
    
    def fetch_folder(self, path:str) -> list[Measurement]:
        all_files = os.listdir(path)
        all_files = sorted(all_files)
        pattern = re.compile(r'(?!.*log).*\.txt$', re.IGNORECASE)
        matching_files = [file for file in all_files if pattern.search(file)]
        all_measurements = []
        for file in matching_files:
            if not ("log" in file.lower() or "png" in file.lower()):
                meas = Measurement(path, file)
                all_measurements.append(meas)
        return all_measurements
