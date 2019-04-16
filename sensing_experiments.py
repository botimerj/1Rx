
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import itertools as itt
from tqdm import tqdm

def main():
    
    # RRAM settings

    n_rows = 3
    n_bit = 1
    roff = 1e6
    ron  = 1e4
    rvar_exp = 0.029

    vdiff = 0.1

    goff = 1/roff
    gon  = 1/ron
    glsb = (1/ron - 1/roff)/(2**n_bit-1)
    glist = np.array([goff+glsb*i for i in range(2**n_bit)])

    i_list_samp = []
    NNN = range(500)
    for nnn in tqdm(NNN):
        # Add offset to glist
        glist_var = [1/(10**(np.log10(1/g) + np.random.normal(0,rvar_exp))) for g in glist]

        # Generate all possible current outputs
        cell_comb = list(itt.combinations_with_replacement(glist_var,n_rows))
        row_comb = list(itt.product([0,vdiff],repeat=n_rows))

        i_list = [np.sum(np.array(c)*np.array(r)) for c in cell_comb for r in row_comb]
        #i_list = np.array(list(set(i_list)))
        i_list.sort()
        i_list_samp.append( i_list )

    i_list_samp = np.array(i_list_samp)
    i_list_avg = np.average(i_list_samp,0)
    i_list_std = np.std(i_list_samp,0)
    #print(i_list_avg)
    #print(i_list_std)

    imin = np.min(i_list)
    imax = np.max(i_list)

    # Reference generation
    adc = ADC(n_bit, n_rows, ron, roff, rvar_exp, vdiff)
    #ilsb = glsb*vdiff
    #i_ref = np.array([ilsb*i for i in range(2**N-1)])+ilsb/2
    #iin_ideal = np.array([0, ilsb, 2*ilsb, 3*ilsb]) 


    # Plot number line
    fig, ax = plt.subplots(1)
    ax.get_yaxis().set_ticks([])
    ax.set_ylim(0, 8)
    ax.set_xlabel('Iout (uA)')
    plt.title('RRAM Read Currents')
    

    for i in i_list_avg:
        plt.vlines(i*1e6, 2, 5)
        #plt.text(i, 3, '{:.2f}'.format(i*1e6), horizontalalignment='center')

    for i in adc.i_ref:
        plt.vlines((i-adc.i_off)*1e6, 2, 7, 'r')

    i_max = 0
    i_min = 0
    for i in range(len(i_list_avg)):
        mu = i_list_avg[i]
        sig = i_list_std[i]
        x = np.linspace(mu-6*sig, mu+6*sig, 100)
        y = mlab.normpdf(x, mu, sig)
        plt.plot(x*1e6, 2*y/np.max(y)+2)

        # Get i_max and i_min
        if np.min(x) < i_min:
            i_min = np.min(x)
        if np.max(x) > i_max:
            i_max = np.max(x)

    i_range = i_max-i_min
    ax.set_xlim((i_min-i_range*0.1)*1e6,(i_max+i_range*0.1)*1e6)
    plt.hlines(2, i_min*1e6, i_max*1e6)
    plt.show()

    #B = 20
    #for i in range(i_list_samp.shape[1]):
    #    pts = i_list_samp[:,i]
    #    plt.hist(pts, bins=B)

    #for i in iin_ideal:
    #    plt.vlines(i, 2-1/2, 2+1/2)
    #    plt.text(i, 3, '{:.2f}'.format(i*1e6), horizontalalignment='center')

    #for i in iref:
    #    plt.vlines(i, 2-1/4, 2+1/4)
    #    plt.text(i, 1, '{:.2f}'.format(i*1e6), horizontalalignment='center')


class ADC():
    def __init__(self, n_bit, n_rows, ron, roff, rvar_exp, vdiff):


        self.N = int(n_bit + np.floor(np.log2(n_bit*n_rows)))
        print(self.N)
        glsb = (1/ron - 1/roff)/(2**n_bit-1)
        #self.i_ref = np.array([(2**i-1/2)*glsb*vdiff for i in range(self.N)])
        self.i_ref = np.array([(2**i)*glsb*vdiff for i in range(self.N)])
        self.i_off = glsb*vdiff/2

        self.vdy = 1         # V-dynamic range
        self.rf = 2*self.vdy/(glsb*vdiff*2**self.N)
        print(np.sum(self.i_ref)*self.rf)
        d_vth = 0.01   # Comparator vth mismatch
        
        print(self.i_ref*self.rf)

    def convert(self, i_in):
        v_in = i_in*self.rf
        #print('vin: {:.3f}'.format(v_in))
        i_tot = -self.i_off
        dout = 0
        for i in range(self.N):
            i_tot += self.i_ref[self.N-i-1]
            #print('{:.3f}'.format(self.rf*i_tot))
            if v_in - i_tot*self.rf > 0 :
                #print("1",end='')
                dout += 2**(self.N-1-i)
            else:
                i_tot -= self.i_ref[self.N-i-1]
                #print("0",end='')

        #print("")
        return dout
        
# Test ADC logic
def adc_test():

    n_rows = 3
    n_bit = 2
    roff = 1e6
    ron  = 1e3
    rvar_exp = 0.0001

    vdiff = 0.1
    
    goff = 1/roff
    gon  = 1/ron
    glist = np.linspace(0, n_rows*gon, 1000) 
    #glist = [0, gon, 2*gon, 3*gon]

    adc = ADC(n_bit, n_rows, ron, roff, rvar_exp, vdiff)
    dout = [adc.convert(g*vdiff) for g in glist]
    plt.plot(glist*vdiff*adc.rf, dout)
    plt.show()

# Logarithmic R variance 
def rvar_exp():
    B = 50
    M = 10000

    exp_mu = 3 
    exp_sig = 0.05 
    exp_samp = np.random.normal(exp_mu, exp_sig, M)
    rsamp_low = np.copy(10**exp_samp)

    x_r_low = np.linspace(exp_mu-6*exp_sig, exp_mu+6*exp_sig, 100)
    y_r_low = mlab.normpdf(x_r_low, exp_mu, exp_sig)

    exp_mu = 5 
    exp_sig = 0.05 
    exp_samp = np.random.normal(exp_mu, exp_sig, M)
    rsamp_high = np.copy(10**exp_samp)

    plt.figure()
    plt.subplot(311)
    plt.hist(rsamp_low, bins=B)
    plt.hist(rsamp_high, bins=B)
    plt.xscale('log')

    plt.subplot(312)
    plt.plot(x_r_low, y_r_low)

    plt.subplot(313)
    plt.plot(x_r_low, y_r_low.cumsum())

    plt.show()


# Linear R variance 
def rvar():
    B = 50
    M = 10000
    ron = 1e3
    rvar = 500 

    rsamp = np.random.normal(ron, rvar, M)
    print("rsamp (mean, std): ", [rsamp.mean(), rsamp.std()])
    gsamp = 1/rsamp
    gmu, gvar = [gsamp.mean(), gsamp.std()]
    print("gsamp (mean, std): ", [gmu, gvar])

    print("gsamp (mean, std): ", [1/ron, rvar/(ron*ron)])

    x_r = np.linspace(ron-6*rvar, ron+6*rvar, 100)
    y_r = mlab.normpdf(x_r, ron, rvar)

    x_g = np.linspace(gmu-6*gvar, gmu+6*gvar, 100)
    y_g = mlab.normpdf(x_g, gmu, gvar)

    plt.figure()
    plt.subplot(121)
    plt.hist(rsamp,bins=B,range=[ron-6*rvar, ron+6*rvar], normed=True)
    plt.plot(x_r, y_r)
    plt.subplot(122)
    plt.plot(x_g, y_g)
    plt.hist(gsamp,bins=B,range=[gmu-6*gvar, gmu+6*gvar], normed=True)
    plt.show()



main()
#adc_test()
#rvar()
#rvar_exp()

