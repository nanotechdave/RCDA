from modules.memorycapacity import memorycapacity
from utils import utils as ut
import pandas as pd
import os


class Measurement():
    def __init__(self, path:str, filename: str):
        self.data, self.elec = ut.read_and_parse_to_df(os.path.join(path, filename))
        self.number = filename[0:3]
        self.experiment = ut.experiment_from_filename(filename)
        return

    def __str__(self):
        return(f"Meas.x")
    

class MemoryCapacity(Measurement):
    def __init__(self):
        return
    
