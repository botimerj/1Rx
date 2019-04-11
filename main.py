
import numpy as np
from rram import Rram
from mvm import MVM

from global_parameters import GlobalParameters 

def test_mvm():

    # Default global params
    gp = GlobalParameters()
    gp.

    rram_dim = [16, 32, 64]
    n_bit = [1, 2, 4]
    active_rows = [1, 2, 4]

    for d in rram_dim:
        for n in n_bit:
            for a in active_rows:
                gp.rram.size_x = d
                gp.rram.size_y = d
                gp.rram.n_bit = n 
                gp.mvm.active_rows = a 
                print('{}x{} | {} | {} :'.format(d,d,n,a),end='')

                
                M = 64 
                N = 64 
                res = 8 

                mvm = MVM(gp)
                
                vec = np.random.random([1,M])*2-1
                mat = np.random.random([M,N])*2-1
                
                result = mvm.dot(vec, mat, res)
                result_t = mvm.dot_truth(vec, mat, res)


                if False in (result == result_t):
                    print("Fail")
                else:
                    print("Pass")

def test_rram():
    M = 1 
    N = 2 
    res = 2 


    vec_q = np.random.random([1,M])*2-1
    vec_q = np.array(np.floor(vec_q*(2**(res-1)-0.001)),dtype=int)
    
    mat_q = np.random.random([M,N])*2-1
    mat_q = np.array(np.floor(mat_q*(2**(res-1)-0.001)),dtype=int)

    rram = Rram()
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
    #main()
    #for i in range(10):
    #test_rram()
    test_mvm()
