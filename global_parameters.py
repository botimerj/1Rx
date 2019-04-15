import numpy as np
import sys

import configparser as cp

class RRAM:
    def __init__(self, config, name="RRAM"):
        self.size_x    = int(config.get(name, "size_x"))
        self.size_y    = int(config.get(name, "size_y"))

        self.von       = float(config.get(name, "von"))
        self.voff      = float(config.get(name, "voff"))

        self.ron       = float(config.get(name, "ron"))
        self.roff      = float(config.get(name, "roff"))
        self.rp        = float(config.get(name, "rp"))
        self.rvar      = float(config.get(name, "rvar"))

        self.n_bit     = int(config.get(name, "n_bit"))

class MVM:
    def __init__(self, config, name="MVM"):
        self.active_rows = int(config.get(name, "active_rows"))

class ADC:
    def __init__(self, config, name="ADC"):
        self.comp_var = float(config.get(name, "comp_var"))


class GlobalParameters:

    def __init__(self):
        config = cp.ConfigParser()
        config.read("config.ini")
        self.rram = RRAM(config)
        self.mvm  = MVM(config)
        self.adc  = ADC(config)


        # Calculated parameters
        self.mvm.adc_res = self.rram.n_bit +\
                           np.ceil(np.log2(self.mvm.active_rows))
