import numpy as np

class ADC():
    def __init__(self, n_bit, n_rows, ron, roff, rvar_exp, vdiff):
        self.N = int(n_bit + np.floor(np.log2(n_bit*n_rows)))
        glsb = (1/ron - 1/roff)/(2**n_bit-1)
        self.i_ref = np.array([(2**i)*glsb*vdiff for i in range(self.N)])
        self.i_off = glsb*vdiff/2

        self.vdy = 1         # V-dynamic range
        self.rf = 2*self.vdy/(glsb*vdiff*2**self.N)
        d_vth = 0.01   # Comparator vth mismatch
        

    def convert(self, i_in):
        v_in = i_in*self.rf
        i_tot = -self.i_off
        dout = 0
        for i in range(self.N):
            i_tot += self.i_ref[self.N-i-1]
            if v_in - i_tot*self.rf > 0 :
                dout += 2**(self.N-1-i)
            else:
                i_tot -= self.i_ref[self.N-i-1]
        return int(dout)

