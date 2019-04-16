
import numpy as np
import time
from rram import RRAM
from mvm import MVM

from global_parameters import GlobalParameters 

def test_mvm():
    # Default global params
    gp = GlobalParameters()

    dim = [64]
    n_bit = [1]
    #active_rows = [1, 2, 3, 8, 16, 20, 24, 28, 32]
    #active_rows = [1, 2, 8, 16, 32, 64]
    active_rows = [1, 2, 4]
    settings = [[d,n,a] for d in dim for n in n_bit for a in active_rows]

    print('R-Dim | Nb | AR')
    print('---------------')
    for s in settings:
        gp.rram.size_x = s[0]
        gp.rram.size_y = s[0]
        gp.rram.n_bit  = s[1] 
        gp.mvm.active_rows =  s[2]
        print('{}x{} | {} | {} :'.format(s[0],s[0],s[1],s[2]),end='')

        
        M = 64 
        N = 64 
        res = 8 

        mvm = MVM(gp)
        
        vec = np.random.random([1,M])*2-1
        mat = np.random.random([M,N])*2-1
        
        start = time.time()
        result = mvm.dot(vec, mat, res)
        print('{:.2f}'.format(mvm.e_read*1e12), end=' ')

        result_t = mvm.dot_truth(vec, mat, res)

        if False in (result == result_t):
            print("Fail")
        else:
            print("Pass")

        #start = time.time()
        #print('{:.2f}'.format((time.time()-start)*1e3), end=' ')


def energy_graphs():
    gp = GlobalParameters()
    rram = RRAM(gp)
    rram.adc.energy_calc(plot=True)
    print("ADC resolution: ", rram.adc.N)

def test_rram():
    M = 1 
    N = 2 
    res = 2 


    vec_q = np.random.random([1,M])*2-1
    vec_q = np.array(np.floor(vec_q*(2**(res-1)-0.001)),dtype=int)
    
    mat_q = np.random.random([M,N])*2-1
    mat_q = np.array(np.floor(mat_q*(2**(res-1)-0.001)),dtype=int)

    rram = RRAM()
    rram.write(mat_q, res)

    bias_reg =  np.sum(mat_q,0)*2**(res-1)
    bias_reg += np.sum(vec_q)*2**(res-1)
    bias_reg += (2**(res-1))**2*vec_q.shape[1]

    out = rram.read(vec_q,res)
    out = out - bias_reg

    print(out)
    print(np.dot(vec_q, mat_q))

    if False in (out == np.dot(vec_q, mat_q)):
        print("Fail")
    else:
        print("Pass")
    

if __name__=="__main__":
    #test_mvm()
    energy_graphs()
