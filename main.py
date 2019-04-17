
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
import copy

from rram import RRAM
from mvm import MVM
from global_parameters import GlobalParameters 

def test_mvm():
    # Default global params
    gp = GlobalParameters()

    dim = [64]
    n_bit = [1]
    #active_rows = [1, 2, 3, 8, 16, 20, 24, 28, 32]
    active_rows = [1, 2, 8, 16, 32, 64]
    #active_rows = [1, 15, 63, 128]
    settings = [[d,n,a] for d in dim for n in n_bit for a in active_rows]

    print('R-Dim | Nb | AR')
    print('---------------')
    for s in settings:
        gp.rram.size_x = s[0]
        gp.rram.size_y = s[0]
        gp.rram.n_bit  = s[1] 
        gp.mvm.active_rows =  s[2]
        print('{}x{} | {} | {} :'.format(s[0],s[0],s[1],s[2]),end='')

        
        M = 128 
        N = 128 
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

def adc_energy_graphs():
    gp = GlobalParameters()
    rram = RRAM(gp)
    rram.adc.energy_calc(plot=True)
    rram.adc.energy_calc()
    print("Conversion Energy: ", rram.adc.energy)
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
    
def boundary_test():
    M = 4 
    N = 4 
    res = 8 
    
    gp = GlobalParameters()
    mvm = MVM(gp)
    
    vec = np.random.random([1,M])*2-1
    print("==Vec==")
    print(vec)
    mat = np.random.random([M,N])*2-1
    print("==Mat==")
    print(mat)
    
    result = mvm.dot(vec, mat, res)
    print("==Res==")
    print(result)
    result_t = mvm.dot_truth(vec, mat, res)
    
    if False in (result == result_t):
        print("Fail")
    else:
        print("Pass")

def rratio_arows_shmoo():
    # Default global params
    gp = GlobalParameters()
    gp.rram.rvar = 0
    gp.rram.size_x = 256 
    gp.rram.size_y = 256 

    #r_var = np.log10([1, 1.01, 1.05, 1.1, 1.2, 1.5, 2, 3])/6
    comp_var = [0.000, 0.002, 0.004, 0.008, 0.016, 0.032]
    active_rows = [1, 3, 7, 15, 31, 63, 127, 255]
    #r_var = np.log10([1, 1.01, 2])/6
    #active_rows = [1, 15, 63]

    settings = [[c,a] for c in comp_var for a in active_rows]

    shmoo_grid = []
    for s in settings:
        gp.adc.comp_var = s[0] 
        gp.mvm.active_rows =  s[1]
        print('{} | {} :'.format(s[0],s[1]),end='')

        
        M = 128 
        N = 32 
        res = 8 

        mvm = MVM(gp)
        
        vec = np.random.random([1,M])*2-1
        mat = np.random.random([M,N])*2-1
        
        start = time.time()
        result = mvm.dot(vec, mat, res)
        print('{:.2f}:'.format(mvm.e_read*1e12), end=' ')

        result_t = mvm.dot_truth(vec, mat, res)

        if False in (result == result_t):
            shmoo_grid.append(0)
            print("Fail")
        else:
            shmoo_grid.append(1)
            print("Pass")

    print(shmoo_grid)

    shmoo_grid = np.array(shmoo_grid).reshape(len(comp_var),len(active_rows))
    print(shmoo_grid)

    f = open("outputs/compv_arows_schmoo", 'w')
    f.write(str(shmoo_grid))
    f.close()
def compv_arows_shmoo():
    # Default global params
    gp = GlobalParameters()
    gp.rram.rvar = 0
    gp.rram.size_x = 256 
    gp.rram.size_y = 256 

    #r_var = np.log10([1, 1.01, 1.05, 1.1, 1.2, 1.5, 2, 3])/6
    comp_var = [0.000, 0.002, 0.004, 0.008, 0.016, 0.032]
    active_rows = [1, 3, 7, 15, 31, 63, 127, 255]
    #r_var = np.log10([1, 1.01, 2])/6
    #active_rows = [1, 15, 63]

    settings = [[c,a] for c in comp_var for a in active_rows]

    shmoo_grid = []
    for s in settings:
        gp.adc.comp_var = s[0] 
        gp.mvm.active_rows =  s[1]
        print('{} | {} :'.format(s[0],s[1]),end='')

        
        M = 128 
        N = 32 
        res = 8 

        mvm = MVM(gp)
        
        vec = np.random.random([1,M])*2-1
        mat = np.random.random([M,N])*2-1
        
        start = time.time()
        result = mvm.dot(vec, mat, res)
        print('{:.2f}:'.format(mvm.e_read*1e12), end=' ')

        result_t = mvm.dot_truth(vec, mat, res)

        if False in (result == result_t):
            shmoo_grid.append(0)
            print("Fail")
        else:
            shmoo_grid.append(1)
            print("Pass")

    print(shmoo_grid)

    shmoo_grid = np.array(shmoo_grid).reshape(len(comp_var),len(active_rows))
    print(shmoo_grid)

    f = open("outputs/compv_arows_schmoo", 'w')
    f.write(str(shmoo_grid))
    f.close()

def rvar_arows_shmoo():
    # Default global params
    gp = GlobalParameters()
    gp.adc.comp_var = 0.000
    gp.rram.size_x = 256 
    gp.rram.size_y = 256 

    r_var = np.log10([1, 1.01, 1.05, 1.1, 1.2, 1.5, 2, 3])/6
    active_rows = [1, 3, 7, 15, 31, 63, 127, 255]
    #r_var = np.log10([1, 1.01, 2])/6
    #active_rows = [1, 15, 63]

    settings = [[r,a] for r in r_var for a in active_rows]
    shmoo_grid = []

    print('R-Dim | Nb | AR')
    print('---------------')
    
    shmoo_grid = []
    for s in settings:
        gp.rram.rvar  = s[0] 
        gp.mvm.active_rows =  s[1]
        print('{} | {} :'.format(10**(6*s[0]),s[1]),end='')

        
        M = 128 
        N = 32 
        res = 8 

        mvm = MVM(gp)
        
        vec = np.random.random([1,M])*2-1
        mat = np.random.random([M,N])*2-1
        
        start = time.time()
        result = mvm.dot(vec, mat, res)
        print('{:.2f}:'.format(mvm.e_read*1e12), end=' ')

        result_t = mvm.dot_truth(vec, mat, res)

        if False in (result == result_t):
            shmoo_grid.append(0)
            print("Fail")
        else:
            shmoo_grid.append(1)
            print("Pass")

    shmoo_grid = np.array(shmoo_grid).reshape(len(r_var),len(active_rows))
    print(shmoo_grid)

    f = open("outputs/schmoo", 'w')
    f.write(str(shmoo_grid))
    f.close()


def energy_vs_adc_res():
    # Default global params
    gp = GlobalParameters()
    
    # Define Constant Global params
    gp.adc.comp_var = 0.000
    gp.rram.size_x = 64 
    gp.rram.size_y = 64 
    gp.rram.r_var = np.log10(1)/6

    #Define Parametric Global params
    set1 = [1,2,4,8]
    set2 = [1, 3, 7, 15] 
    #set2 = [1, 3] 

    settings = [[s1,s2] for s1 in set1 for s2 in set2]
    gp_list = []
    for s in settings:
        gp.rram.n_bit  = s[0] 
        gp.mvm.active_rows =  s[1]
        gp_list.append(copy.deepcopy(gp))

    [distance, energy, bools] = sweep_gp_params(gp_list)
    print(np.array(energy))
    energy = np.array(energy)*1e12/(128*128+127*128)

    f = open("outputs/energy_vs_adc_res", 'w')
    for e in energy:
        f.write(str(energy))
    f.close()


def distance_vs_active_rows():
    # Default global params
    gp = GlobalParameters()
    
    # Define Constant Global params
    gp.adc.comp_var = 0.005
    gp.rram.size_x = 128
    gp.rram.size_y = 128 
    #gp.rram.size_x = 8 
    #gp.rram.size_y = 8 

    #Define Parametric Global params
    set1 = np.log10([1, 1.05, 1.1, 1.5, 2])/6
    set2 = [1, 3, 7, 15, 31, 63, 127] 
    #set1 = np.log10([1, 2])/6
    #set2 = [1, 3, 7] 

    settings = [[s1,s2] for s1 in set1 for s2 in set2]
    gp_list = []
    for s in settings:
        gp.rram.rvar  = s[0] 
        gp.mvm.active_rows =  s[1]
        gp_list.append(copy.deepcopy(gp))


    [distance, energy, bools] = sweep_gp_params(gp_list)
    distance = np.array(distance).reshape(len(set1),len(set2))
    #print(distance)


    f = open("outputs/distance_vs_arows", 'w')
    for d in distance:
        f.write(str(d)+'\n')
    f.close()
    
    fig, ax = plt.subplots()
    #ax.set_xlabel('Active Rows (K)')
    ax.set_xlabel('ADC Resolution (N)')
    ax.set_ylabel('Accuracy (Distance from truth)')
    ax.set_title('MAC accuracy')

    for i in range(distance.shape[0]):
        plt.plot(np.log2(np.array(set2)+1), distance[i])

    r_var = (10**(np.array(set1)*6)-1)*100
    r_var_str = ['{:.0f}%'.format(r) for r in r_var]
    ax.legend(r_var_str, title='Rcell Variation')
    plt.show()


def energy_vs_active_rows():
    # Default global params
    gp = GlobalParameters()
    
    # Define Constant Global params
    gp.adc.comp_var = 0.000
    gp.rram.size_x = 256 
    gp.rram.size_y = 256 
    gp.rram.r_var = np.log10(1)/6

    #Define Parametric Global params
    set1 = [0]
    set2 = [1, 3, 7, 15, 31, 63, 127, 255] 
    #set2 = [1, 3] 

    settings = [[s1,s2] for s1 in set1 for s2 in set2]
    gp_list = []
    for s in settings:
        gp.rram.rvar  = s[0] 
        gp.mvm.active_rows =  s[1]
        gp_list.append(copy.deepcopy(gp))

    [distance, energy, bools] = sweep_gp_params(gp_list)
    print(np.array(energy))
    energy = np.array(energy)*1e12/(128*128+127*128)

    f = open("outputs/energy_vs_arows", 'w')
    for e in energy:
        f.write(str(energy))
    f.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('ADC Resolution (N)')
    ax.set_ylabel('Energy/OP (pJ)')
    ax.set_title('Energy Efficiency of MAC')
    plt.plot(np.log2(np.array(set2)+1), energy)
    plt.show()


def sweep_gp_params(gp_list):
    distance_list = []
    energy_list   = []
    bool_list     = []

    print('===Sweeping Params===')
    for gp_i in gp_list:

        M = 128 
        N = 128 
        res = 8 

        mvm = MVM(gp_i)
        
        vec = np.random.random([1,M])*2-1
        mat = np.random.random([M,N])*2-1
        
        start = time.time()
        result = mvm.dot(vec, mat, res)
        #print('{:.2f}'.format(mvm.e_read*1e12), end=' ')
        result_t = mvm.dot_truth(vec, mat, res)
        energy_list.append(mvm.e_read)

        if False in (result == result_t):
            bool_list.append(0)
            print("Fail")
        else:
            bool_list.append(1)
            print("Pass")

        max_val = np.max(result_t)
        distance_list.append(np.copy(np.sum(np.abs(result/max_val-result_t/max_val))/np.size(result)))

    return distance_list, energy_list, bool_list

if __name__=="__main__":
    #distance_vs_active_rows()
    #energy_vs_active_rows()
    #energy_vs_adc_res()
    #test_mvm()
    adc_energy_graphs()
    #boundary_test()
    #rvar_arows_shmoo()
    #compv_arows_shmoo()
