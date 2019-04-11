
import numpy as np
from rram import Rram

from global_parameters import GlobalParameters
gp = GlobalParameters()

class MVM():
    def __init__(self):
        # Energy variables
        self.read_energy = 0
        self.write_energy = 0

        # Mapping parameters
        self.rram_arr_size = [0,0]
        self.compute_steps = 0
        self.adc_res = 0

    def dot_truth(self, vec, mat, res):

        # Quantize inputs
        mat_q = np.floor(np.copy(mat)*(2**(res-1)-0.001))
        vec_q = np.floor(np.copy(vec)*(2**(res-1)-0.001))

        return np.dot(vec_q, mat_q)

    def dot(self, vec, mat, res):

        # Calculate parameters for mapping RRAM
        w_per_rram_x = int(gp.rram.size_x/np.floor(res/gp.rram.n_bit))
        self.rram_arr_size = [int(np.ceil(mat.shape[0]/gp.rram.size_y)),
                              int(np.ceil(mat.shape[1]/w_per_rram_x))]

        print("vec in shape: ", vec.shape)
        print("mat in shape: ", mat.shape)
        print("RRAM arr shape: ", self.rram_arr_size)


        self.compute_steps = int(gp.rram.size_y/gp.mvm.active_rows)
        self.adc_res = gp.rram.n_bit + np.log2(gp.mvm.active_rows)

        # Quantize inputs
        mat_q = np.floor(np.copy(mat)*(2**(res-1)-0.001))
        vec_q = np.floor(np.copy(vec)*(2**(res-1)-0.001))
        
        # Calc bias offset 
        bias_reg =  np.sum(mat_q,0)*2**(res-1)
        bias_reg += np.sum(vec_q)*2**(res-1)
        bias_reg += (2**(res-1))**2*vec.shape[1]

        mat_q += 2**(res-1)
        vec_q += 2**(res-1)



        # Create RRAMs and load with data
        rram_arr = [ [Rram() for j in range(self.rram_arr_size[1])] 
                             for i in range(self.rram_arr_size[0]) ]
        
        for i in range(self.rram_arr_size[0]): 
            for j in range(self.rram_arr_size[1]):
                yidx = int(i*gp.rram.size_y)
                xidx = int(j*w_per_rram_x)
                sub_mat = mat_q[yidx:yidx+gp.rram.size_y,\
                                xidx:xidx+w_per_rram_x]
                rram_arr[i][j].write(sub_mat, res)


        # Compute on RRAM
        # Initialize result registers
        result = np.zeros([self.rram_arr_size[1]*gp.rram.size_x])
        
        # Compute over each rram
        for i in range(self.rram_arr_size[0]): 
            for j in range(self.rram_arr_size[1]):
                # Only activate gp.mvm.active_rows at a time 
                for k in range(self.compute_steps):
                    mask = np.zeros(gp.rram.size_y)
                    mask[k*gp.mvm.active_rows:(k+1)*gp.mvm.active_rows] = 1
                    vec_in = np.zeros(gp.rram.size_y)
                    v_tmp = vec_q[0][i*gp.rram.size_y:(i+1)*gp.rram.size_y]
                    vec_in[:v_tmp.shape[0]] = v_tmp
                    vec_in = vec_in*mask
                    a = rram_arr[i][j].read(vec_in, res)
                    result[j*w_per_rram_x:(j+1)*w_per_rram_x] += a.squeeze()

        result = result[:mat_q.shape[1]]

        result = result - bias_reg

        #return np.dot(vec, mat)
        return result

    def print_stats(self):
        print("===Mapping Params===")
        print("rram_arr_size: ", self.rram_arr_size)
        print("compute_steps: ", self.compute_steps)
        print("adc_res      : ", self.adc_res)



