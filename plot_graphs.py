import numpy as np
import matplotlib.pyplot as plt


def compv_arows_shmoo():
    data = [[1,1,1,1,1,1]
           ,[1,1,1,1,1,1]
           ,[1,1,1,1,1,0]
           ,[1,1,1,1,1,0]
           ,[1,1,1,1,0,0]
           ,[1,1,0,0,0,0]]



    comp_var = [0.000, 0.002, 0.004, 0.008, 0.016, 0.032]
    comp_var_ticks = np.array(comp_var)*1000
    active_rows = [7, 15, 31, 63, 127, 255]

    fig, ax = plt.subplots()

    # Major ticks
    ax.set_xticks(np.arange(0,6,1))
    ax.set_yticks(np.arange(0,6,1))
    
    ax.set_xticklabels(active_rows)
    ax.set_yticklabels(comp_var_ticks)
    
    # Minor ticks
    ax.set_xticks(np.arange(-0.5,6,1), minor=True)
    ax.set_yticks(np.arange(-0.5,6,1), minor=True)

    ax.grid(which='minor', color='black', linewidth=2, linestyle='-')

    ax.set_xlabel('Activated Rows')
    ax.set_ylabel('Comparator Variation (mV)')
    ax.set_title('RRAM Computation Accuracy (Pass/Fail)')

    plt.imshow(data)
    plt.show()

def rvar_arows_shmoo():
    #data = np.array([[1,1,1,1,1,1,1,1], 
    #                 [1,1,1,1,1,1,1,0],
    #                 [1,1,1,1,1,1,1,0],
    #                 [1,1,1,1,1,1,1,0],
    #                 [1,1,1,1,1,0,0,0],
    #                 [1,1,0,0,0,0,0,0],
    #                 [1,0,0,0,0,0,0,0],
    #                 [0,0,0,0,0,0,0,0]])

    data = [[1,1,1,1,1,1,1,1]
           ,[1,1,1,1,1,1,1,1]
           ,[1,1,1,1,1,1,1,1]
           ,[1,1,1,1,1,1,1,1]
           ,[1,1,1,1,1,1,0,0]
           ,[1,1,1,0,0,0,0,0]
           ,[1,0,0,0,0,0,0,0]
           ,[1,0,0,0,0,0,0,0]]

    r_var = (np.array([1, 1.01, 1.05, 1.1, 1.2, 1.5, 2, 3])-1)*100
    r_var_ticks = ['{:.0f}%'.format(r) for r in r_var]
    active_rows = [1, 3, 7, 15, 31, 63, 127, 255]

    fig, ax = plt.subplots()

    # Major ticks
    ax.set_xticks(np.arange(0,8,1))
    ax.set_yticks(np.arange(0,8,1))
    
    ax.set_xticklabels(active_rows)
    ax.set_yticklabels(r_var_ticks)
    
    # Minor ticks
    ax.set_xticks(np.arange(-0.5,8,1), minor=True)
    ax.set_yticks(np.arange(-0.5,8,1), minor=True)

    ax.grid(which='minor', color='black', linewidth=2, linestyle='-')

    ax.set_xlabel('Activated Rows')
    ax.set_ylabel('R Variation')
    ax.set_title('RRAM Computation Accuracy (Pass/Fail)')

    plt.imshow(data)
    plt.show()

#rvar_arows_shmoo()
compv_arows_shmoo()

def energy_vs_adc_res():
    pass
