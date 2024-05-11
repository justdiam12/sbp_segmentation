# This is the code to simulate the SBP data
import numpy as np
import matplotlib.pyplot as plt

class SBP_Simulate():
    def __init__(self, layers, rho, c):
        self.layers = layers
        self.rho = rho
        self.c = c
        self.rho_profile = np.zeros((256, 1))
        self.c_profile = np.zeros((256, 1))
        self.twtt = np.array([], dtype=np.float32)

        for i in range(len(self.layers)):
            if i == 0:
                self.rho_profile[0:self.layers[i]] = self.rho[i]
                self.c_profile[0:self.layers[i]] = self.c[i]
                self.rho_profile[self.layers[i]:self.layers[i+1]] = self.rho[i+1]
                self.c_profile[self.layers[i]:self.layers[i+1]] = self.c[i+1]
            elif i == len(layers)-1:
                self.rho_profile[self.layers[i]:] = self.rho[i+1]
                self.c_profile[self.layers[i]:] = self.c[i+1]
            else:
                self.rho_profile[self.layers[i]:self.layers[i+1]] = self.rho[i+1]
                self.c_profile[self.layers[i]:self.layers[i+1]] = self.c[i+1]
        self.impedance = self.rho_profile * self.c_profile
        
    def simulate(self):
        start_index = 0
        up_down = -1 
        self.pulse(start_index, up_down)
        # R12 = (self.rho[1]*self.c[1] - self.rho[0]*self.c[0]) / (self.rho[1]*self.c[1] + self.rho[0]*self.c[0])
        # T12 = (2*self.rho[1]*self.c[1]) / (self.rho[1]*self.c[1] + self.rho[0]*self.c[0])
        # R23 = T12 * ((self.rho[2]*self.c[2] - self.rho[1]*self.c[1]) / (self.rho[2]*self.c[2] + self.rho[1]*self.c[1]))
        # T21 = R23 * (2*self.rho[0]*self.c[0]) / (self.rho[0]*self.c[0] + self.rho[1]*self.c[1])
        # print(R12)
        # print(T21)
        # self.profile[101] = R12
        # self.profile[161] = T21

    def pulse(self, start_index, up_down):
        return

def run():
    layers = [10, 50, 60]
    rho = [1026, 1906, 1922, 1955]
    c = [1500, 1600, 1650, 1660]
    sbp_simulate = SBP_Simulate(layers, rho, c)
    print(sbp_simulate.impedance)


if __name__ == '__main__':
    run()