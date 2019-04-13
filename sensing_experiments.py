
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import itertools as itt
from tqdm import tqdm

def main():

    # ADC settings
    N = 2           # ADC resolution

    vdy = 1         # V-dynamic range
    vth = 0.3       # Threshold voltage
    vm = vth+vdy

    d_vth = 0.001   # Comparator vth mismatch
    
    # RRAM settings

    n_rows = 8
    n_bit = 1
    roff = 1e5
    ron  = 1e3
    rvar_exp = 0.002

    vdiff = 0.1

    goff = 1/roff
    gon  = 1/ron
    glsb = (1/ron - 1/roff)/(2**n_bit-1)
    glist = np.array([goff+glsb*i for i in range(2**n_bit)])

    i_list_samp = []
    NNN = range(200)
    for nnn in tqdm(NNN):
        # Add offset to glist
        glist_var = [1/(10**(np.log10(1/g) + np.random.normal(0,rvar_exp))) for g in glist]

        # Generate all possible current outputs
        cell_comb = list(itt.combinations_with_replacement(glist_var,n_rows))
        row_comb = list(itt.product([0,1],repeat=n_rows))

        i_list = [np.sum(np.array(c)*np.array(r)) for c in cell_comb for r in row_comb]
        #i_list = np.array(list(set(i_list)))
        i_list.sort()
        i_list_samp.append( i_list )

    i_list_samp = np.array(i_list_samp)
    i_list_avg = np.average(i_list_samp,0)
    i_list_std = np.std(i_list_samp,0)
    print(i_list_avg)
    print(i_list_std)

    imin = np.min(i_list)
    imax = np.max(i_list)

    # Reference generation
    ilsb = glsb*vdiff
    iref = np.array([ilsb*i for i in range(2**N-1)])+ilsb/2
    print(iref)
    #iin_ideal = np.array([0, ilsb, 2*ilsb, 3*ilsb]) 


    # Plot number line
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    irange = imax-imin
    ax.set_xlim(imin-irange*0.1,imax+irange*0.1)
    ax.set_ylim(0, 10)
    
    plt.hlines(2, imin, imax)

    for i in i_list:
        plt.vlines(i, 2, 5)
        #plt.text(i, 3, '{:.2f}'.format(i*1e6), horizontalalignment='center')

    for i in range(len(i_list_avg)):
        mu = i_list_avg[i]
        sig = i_list_std[i]
        x = np.linspace(mu-6*sig, mu+6*sig, 100)
        y = mlab.normpdf(x, mu, sig)
        plt.plot(x, 2*y/np.max(y)+2)

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
#rvar()
#rvar_exp()

