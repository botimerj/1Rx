
import sys
import numpy as np

from global_parameters import GlobalParameters
gp = GlobalParameters()


class Rram:
    
    def __init__(self):

        self.ron  = gp.rram.ron
        self.roff = gp.rram.roff
        self.rvar = gp.rram.rvar
        self.rp   = gp.rram.rp

        self.n_bit = gp.rram.n_bit
        self.rlsb = (self.roff-self.ron)/(2**self.n_bit-1)
        self.glsb = (1/self.ron - 1/self.roff)/(2**self.n_bit-1)

        self.x = gp.rram.size_x
        self.y = gp.rram.size_y

        # Resistance values
        self.arr = np.empty([self.y, self.x])

        # Digital values
        self.dig_arr = np.empty([self.y, self.x])

    def write(self, weights, res):
        # Helper variables
        n_bit = int(self.n_bit)
        n_cell = int(np.ceil(res/n_bit))
        if(n_cell > self.x):
            raise Exception("No weight splitting allowed")
        
        w = np.array(weights,dtype=int)

        # Generate digital represntation in weight arr
        self.dig_arr = np.zeros([self.y, self.x])
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                try:
                    num = int(w[i][j])
                    a = [(num>>(n_bit*i))&(2**n_bit-1) for i in range(n_cell)]
                    a = np.flip(np.array(a,dtype=int))
                    self.dig_arr[i][j*n_cell:(j+1)*n_cell] = a
                except:
                    print("except")
                    pass

        # Assign real resistances to r_cell 
        for i in range(self.y): 
            for j in range(self.x): 
                self.arr[i][j]  = 1/self.roff + self.glsb*self.dig_arr[i][j]
                self.arr[i][j]  = 1/(1/self.arr[i][j] + np.random.normal(0,self.rvar))
                #print(1/self.arr[i][j])
    
    def read(self, ifmap, res):

        ifm = np.array(ifmap, dtype=int)

        # Bit-serial approach
        dout = np.zeros([1,self.x])
        for i in range(res):
            v = ((np.copy(ifm)>>i)&1)*(gp.rram.von-gp.rram.voff)
            i_out = np.dot(v, self.arr)
            dout = dout + (self.adc(i_out)<<i)
        dout = np.array(dout,dtype=int)

        # Concatenate columns 
        n_cell = int(np.ceil(res/self.n_bit))
        num_words = int(np.floor(self.x/n_cell))

        out = np.zeros([1,num_words])
        for i in range(num_words):
            for j in range(n_cell):
                idx = i*n_cell+j
                out[0][i] += (dout[0][idx]<<((n_cell - 1 - j)*self.n_bit))

        return out

        

    def adc(self, i_in):
        rows = gp.mvm.active_rows
        vdiff = gp.rram.von-gp.rram.voff
        i_lsb = vdiff*self.glsb

        return np.array(np.floor(i_in/i_lsb),dtype=int)

