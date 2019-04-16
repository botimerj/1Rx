import matplotlib.pyplot as plt
import numpy as np

class ADC():
    def __init__(self, gp, n_bit, n_rows, ron, roff, rvar_exp, vdiff):
        self.N = int(n_bit + np.floor(np.log2(n_bit*n_rows)))

        # Reference generation
        glsb = (1/ron - 1/roff)/(2**n_bit-1)
        self.i_ref = np.array([(2**i)*glsb*vdiff for i in range(self.N)])
        self.i_off = glsb*vdiff/2

        # Comparison variables
        self.vdy = 1                                # V-dynamic range
        self.rf = 2*self.vdy/(glsb*vdiff*2**self.N) # Convert resistor
        self.comp_var = np.random.normal(0, gp.adc.comp_var)   # Comparator vth mismatch

        # Energy calculation
        self.energy = 0
        self.energy_calc(plot=False)

        

    def convert(self, i_in):
        v_in = i_in*self.rf
        i_tot = -self.i_off
        dout = 0
        for i in range(self.N):
            i_tot += self.i_ref[self.N-i-1]
            if v_in - i_tot*self.rf + self.comp_var > 0 :
                dout += 2**(self.N-1-i)
            else:
                i_tot -= self.i_ref[self.N-i-1]
        return int(dout)


    def energy_calc(self, plot=False):
        # sense + e_rram + N*(pre_amp + comparator + shift + accumulation)
        t_sense = t_amp = t_comp = t_logic = 1e-9
        vdd = 1
        e_sense = vdd*10e-6*t_sense # 1V * 10uA * 1ns
        e_rram  = vdd*self.i_ref[0]*t_sense

        e_preamp = vdd*self.i_ref[0]*t_amp
        e_comparator = vdd*10e-6*t_comp

        e_shift = vdd*10e-6*t_logic
        e_accum = vdd*10e-6*t_logic

        self.energy = e_sense + e_rram + self.N*(e_preamp + e_comparator + e_shift + e_accum)

        if plot:
            labels = ['pre-amp+comparator','digital','sense+rram']
            e_slice = np.array([e_preamp+e_comparator, e_shift+e_accum])*self.N
            e_slice = np.append(e_slice,e_sense+e_rram)
            #e_slice = np.append(e_slice,e_rram)
            e_slice = e_slice*1e15
            for (l,e) in zip(labels, e_slice):
                print('{}: {:.2f}'.format(l,e))

            fig1, ax1 = plt.subplots()
            plt.title('ADC Energy Breakdown (N={})'.format(self.N))
            ax1.pie(e_slice, labels=labels, startangle=90, autopct='%1.1f%%')
            ax1.axis('equal')
            plt.show()


