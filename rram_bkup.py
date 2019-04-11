import numpy as np
from random import randint
from global_parameters import GlobalParameters

gp = GlobalParameters()

# RRAM spice model

class rram:
    # X/Y-array size; N-resolution of rram cell
    def __init__(self, dimension = None, n_bit = None):
        # Physical constraints
        self.ron  = gp.rram.ron
        self.roff = gp.rram.roff
        self.rvar = gp.rram.rvar
        self.rp   = gp.rram.rp

        self.n_bit = gp.rram.n_bit
        self.rlsb = (self.roff-self.ron)/(2**self.n_bit-1)

        self.von = gp.rram.von
        self.voff = gp.rram.voff

        # Array size
        if dimension is None:
            self.x = gp.rram.size_x
            self.y = gp.rram.size_y
        else:
            self.x = dimension[0]
            self.y = dimension[1]

        self.arr = np.zeros([self.x, self.y])

    def write(self, weights, offset):
        # Quantize weights and store them in array 
        w = np.copy(weights)
        bins = np.linspace(0,1,2**self.n_bit+1)
        w = np.digitize(w/w.max(),bins,right=True)-1
        self.arr[offset[0]:w.shape[0]+offset[0],offset[1]:w.shape[1]+offset[1]] = w
        for d0 in range(self.arr.shape[0]):
            for d1 in range(self.arr.shape[1]):
                self.arr[d0][d1] = self.ron + self.rlsb*self.arr[d0][d1]

    def read(self, vin):
        # Create system of linear equations for nodal analysis
        nodes = 2*self.y*self.x
        # Node equations
        ne = np.zeros([nodes,nodes]) 
        # Node solutions
        ns = np.zeros([nodes])

        xd1 = self.y
        xd2 = self.x
        yd1 = self.y
        yd2 = self.x

        # Place holder
        g = 1/self.ron
        gp = 1/self.rp

        # Set 'X' Nodes
        for d1 in range(xd1): 
            for d2 in range(xd2): 
                g = 1/self.arr[d1][d2]
                j = d2 + d1*xd2
                y_offset = xd1*xd2

                # Special case, only 1 gp
                if d1 == 0: 
                    idx = d2 + d1*xd2
                    ne[j][idx] = gp+g
                    ne[j][idx+xd2] = -gp
                    ne[j][y_offset+idx] = -g

                # Special case, x constrained 
                elif d1 == xd1-1:
                    idx = d2 + d1*xd2
                    ne[j][idx-xd2] = -gp
                    ne[j][idx] = 2*gp+g
                    ne[j][y_offset+idx] = -g
                    ns[j] = gp*self.voff 

                # Nominal case
                else:
                    idx = d2 + d1*xd2
                    ne[j][idx-xd2] = -gp
                    ne[j][idx] = 2*gp+g
                    ne[j][idx+xd2] = -gp
                    ne[j][y_offset+idx] = -g

        # Set 'Y' equations
        for d1 in range(yd1):
            for d2 in range(yd2):
                j = y_offset + xd2*d1 + d2
                y_offset = xd1*xd2

                # Specials case, y constrained
                if d2 == 0:
                    idx = y_offset + d2 + d1*yd2
                    ne[j][idx] = 2*gp+g
                    ne[j][idx + 1] = -gp
                    ne[j][idx - y_offset] = -g
                    ns[j] = vin[d1]

                # Special case, only 1 gp
                elif d2 == yd2-1:
                    idx = y_offset + d2 + d1*yd2
                    ne[j][idx - 1] = -gp
                    ne[j][idx] = gp+g
                    ne[j][idx - y_offset] = -g

                # Nominal case
                else:
                    idx = y_offset + d2 + d1*yd2
                    ne[j][idx - 1] = -gp
                    ne[j][idx] = 2*gp+g
                    ne[j][idx + 1] = -gp
                    ne[j][idx - y_offset] = -g

        tmp = np.linalg.solve(ne,ns)
        iout = (tmp[(xd1-1)*xd2:xd1*xd2] - self.voff)*gp
        print("Iout : ", iout*1e6, "uA")
        
        return iout 

    def print(self): 
        print("RRAM Array: \n", self.arr)


def main():

    weights = np.random.rand(gp.rram.size_x,gp.rram.size_y)
    # print("weights--\n",weights)
    # print("inputs--\n",inputs)
    # print("outputs--\n",outputs)
    mac_rram =  rram(n_bit=1)

    mac_rram.write(weights, [0,0])
    mac_rram.print()
    
    #vin = np.random.randint(0,2,gp.rram.size_y)
    vin = np.ones(gp.rram.size_y)*gp.rram.von
    print("Vin: \n", vin)
    mac_rram.read(vin)

    #print(weights)
    #print(rramp.arr) 
    #print(rramn.arr) 

    # Perform dot product
    #outputs_rram = np.zeros([M,N])
    #for ni in range(N):
    #    # DAC voltages stimulating array
    #    vin = inputs[:,ni]
    #    Ip = rramp.calc(vin)
    #    In = rramn.calc(vin)
    #    
    #    Vdiff = Ip*Rf - In*Rf
    #    outputs_rram[:,ni] = Vdiff

    #threshold = K*1/Roff*(Von-Voff)
    #outputs_rram[outputs_rram >  threshold] = int(1)
    #outputs_rram[outputs_rram <= threshold] = int(0)
    #print("RRAM Output:\n", outputs_rram)

if __name__=="__main__":
    main()
