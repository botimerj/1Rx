
import numpy as np
from rram import Rram
from mvm import MVM

from global_parameters import GlobalParameters 
gp = GlobalParameters()

def test_mvm():

    M = 4 
    N = 2 
    res = 4

    mvm = MVM()
    
    vec = np.random.random([1,M])*2-1
    mat = np.random.random([M,N])*2-1

    result = mvm.dot(vec, mat, res)
    result_t = mvm.dot_truth(vec, mat, res)

    print(result)
    print(result_t)
    if False in (result == result_t):
        print("Fail")
    else:
        print("Pass")

    #mvm.print_stats()

def test_rram():
    M = 1 
    N = 2 
    res = 8 


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
    test_rram()
    #test_mvm()
