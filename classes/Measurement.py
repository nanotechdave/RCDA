from modules.memorycapacity import memorycapacity
from utils import utils as ut
import pandas as pd
import numpy as np
import os


class Measurement():
    def __init__(self, path:str, filename: str):
        self.data, self.elec = ut.read_and_parse_to_df(os.path.join(path, filename))
        self.number = filename[0:3]
        self.experiment = ut.experiment_from_filename(filename)
        self.filename = filename
        self.path = path
        self.elec_scores = np.zeros(len(self.elec["float"]))
        if self.experiment == "memorycapacity":
            self.compute_elec_scores()
        return

    def __str__(self):
        return(f"Meas. {self.number}, {self.experiment}")
    
    def compute_elec_scores(self):
        
        bias_voltage = []
        gnd_voltage = []
        float_voltage = []
        target_vec = []
        estimated_vec = []

        # fill matrices for ease of computation
        bias_voltage, gnd_voltage, float_voltage = ut.fillVoltageMatFromDf(self.data, self.elec)
        print(f"len voltages: {len(float_voltage)}")
        print(f"len voltages[0]: {len(float_voltage[0])}")
        for idx in range(len(self.elec["float"])):
            self.elec_scores[idx] = 0

        for k in range(30):
            if k!=0:
                # construct the delayed waveform
                bias_voltage_del, float_voltage_del, gnd_voltage_del = memorycapacity.delayWaveforms(bias_voltage, float_voltage, gnd_voltage, k)
                #divide test and train set
                states_train, states_test, target_train, target_test = ut.train_test_split_time_series(
                            float_voltage_del,
                            bias_voltage_del,
                            test_size=0.2,
                            )
                
                features_score = ut.ftest_evaluate(states_train, states_test, target_train) 
                print(features_score)
                for idx, key in enumerate(features_score.keys()):
                    score_series = features_score[key]
                    self.elec_scores[idx] += score_series["F"]                     
        return
    

class MemoryCapacity(Measurement):
    def __init__(self):
        return


