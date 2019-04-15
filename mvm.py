
import numpy as np
from rram import RRAM

class MVM():
    def __init__(self, gp):
        # Global params
        self.gp = gp

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
        # Variables useful for calculation
        rram_x = self.gp.rram.size_x
        rram_y = self.gp.rram.size_y
        n_bit  = self.gp.rram.n_bit
        a_rows = self.gp.mvm.active_rows

        # Calculate parameters for mapping RRAM
        w_per_rram_x = int(rram_x/np.floor(res/n_bit))
        self.rram_arr_size = [int(np.ceil(mat.shape[0]/rram_y)),
                              int(np.ceil(mat.shape[1]/w_per_rram_x))]



        self.compute_steps = int(np.ceil(rram_y/a_rows))
        self.adc_res = n_bit + np.log2(a_rows)

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
        rram_arr = [ [RRAM(self.gp) for j in range(self.rram_arr_size[1])] 
                                    for i in range(self.rram_arr_size[0]) ]
        
        for i in range(self.rram_arr_size[0]): 
            for j in range(self.rram_arr_size[1]):
                yidx = int(i*rram_y)
                xidx = int(j*w_per_rram_x)
                sub_mat = mat_q[yidx:yidx+rram_y,\
                                xidx:xidx+w_per_rram_x]
                rram_arr[i][j].write(sub_mat, res)


        # Compute on RRAM
        # Initialize result registers
        result = np.zeros([self.rram_arr_size[1]*rram_x])
        
        # Compute over each rram
        for i in range(self.rram_arr_size[0]): 
            for j in range(self.rram_arr_size[1]):
                # Only activate a_rows at a time 
                for k in range(self.compute_steps):
                    # Generate input mask (a_row hot encoded)
                    mask = np.zeros(rram_y)
                    mask[k*a_rows:(k+1)*a_rows] = 1

                    # Create correct sized input vector 
                    vec_in = np.zeros(rram_y)
                    v_tmp = vec_q[0][i*rram_y:(i+1)*rram_y]
                    vec_in[:v_tmp.shape[0]] = v_tmp
                    vec_in = vec_in*mask
                    
                    # Perform rram mac
                    a = rram_arr[i][j].read(vec_in, res)
                    # Accumulate result
                    result[j*w_per_rram_x:(j+1)*w_per_rram_x] += a.squeeze()

        # Truncate unsed registers
        result = result[:mat_q.shape[1]]

        # Add in bias
        result = result - bias_reg
        return result

    def print_stats(self):
        print("===Mapping Params===")
        print("rram_arr_size: ", self.rram_arr_size)
        print("compute_steps: ", self.compute_steps)
        print("adc_res      : ", self.adc_res)



