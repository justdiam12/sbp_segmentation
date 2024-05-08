# This is the code to simulate the SBP data
import numpy as np
import matplotlib.pyplot as plt

class SBP_Simulate():
    def __init__(self, layers, rho, c):
        self.layers = layers
        self.rho = rho
        self.c = c
        self.profile = np.zeros((256, 1))
        self.seafloor = np.zeros((256, 256))
        self.rho_profile = np.zeros((256, 1))
        self.c_profile = np.zeros((256, 1))
        self.rho_profile[:100] = rho[0]
        self.rho_profile[101:160] = rho[1]
        self.rho_profile[161:] = rho[2]
        self.c_profile[:100] = c[0]
        self.c_profile[101:160] = c[1]
        self.c_profile[161:] = c[2]
        
    def simulate(self):
        R12 = (self.rho[1]*self.c[1] - self.rho[0]*self.c[0]) / (self.rho[1]*self.c[1] + self.rho[0]*self.c[0])
        T12 = (2*self.rho[1]*self.c[1]) / (self.rho[1]*self.c[1] + self.rho[0]*self.c[0])
        R23 = T12 * ((self.rho[2]*self.c[2] - self.rho[1]*self.c[1]) / (self.rho[2]*self.c[2] + self.rho[1]*self.c[1]))
        T21 = R23 * (2*self.rho[0]*self.c[0]) / (self.rho[0]*self.c[0] + self.rho[1]*self.c[1])
        print(R12)
        print(T21)
        self.profile[101] = R12
        self.profile[161] = T21

    def floor(self):
        for i in range(256):
            self.seafloor[:,i] = self.profile[:,0]
    

def run():
    layers = 3
    rho = [1026, 1906, 1922]
    c = [1500, 1600, 1650]
    sbp_simulate = SBP_Simulate(3, rho, c)
    sbp_simulate.simulate()
    sbp_simulate.floor()

    plt.figure(1)
    plt.plot(sbp_simulate.profile)

    plt.figure(2)
    plt.imshow(255 - sbp_simulate.seafloor*255, cmap='gray')
    plt.show()


if __name__ == '__main__':
    run()