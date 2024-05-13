# This is the code to simulate the SBP data
import numpy as np
import matplotlib.pyplot as plt

class SBP_Simulate():
    def __init__(self, layers, rho, c, m_per_p, thresh):
        self.layers = layers
        self.rho = rho
        self.c = c
        self.m_per_p = m_per_p
        self.thresh = thresh
        self.profile = np.zeros((256,1))
        self.rho_profile = np.zeros((256, 1))
        self.c_profile = np.zeros((256, 1))
        self.twtt = np.array([], dtype=np.float32)
        self.amp = np.array([], dtype=np.float32)
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
        index = 0
        up_down = -1 
        time = 0
        coeff = 1
        self.pulse(index, up_down, time, coeff)

    def pulse(self, index, up_down, time, coeff):
        if up_down == -1: # Descending Pulse
            if index+1 == len(self.profile):
                return
            else:
                while self.impedance[index] == self.impedance[index+1]:
                    time += self.m_per_p / self.c_profile[index]
                    index += 1
                    if index+1 == len(self.profile):
                        return
                R = ((self.impedance[index+1] - self.impedance[index]) / (self.impedance[index+1] + self.impedance[index])) * coeff
                T = ((2*self.impedance[index+1]) / (self.impedance[index+1] + self.impedance[index])) * coeff
                if R >= self.thresh:
                    self.pulse(index, 1, time, R)
                if T >= self.thresh:
                    self.pulse(index+1, -1, time, T)

        elif up_down == 1: # Ascending Pulse
            if index-1 == -1:
                return
            else:
                while self.impedance[index] == self.impedance[index-1]:
                    time += self.m_per_p / self.c_profile[index]
                    index -= 1
                    if index-1 == -1:
                        time += self.m_per_p / self.c_profile[index]
                        self.twtt = np.append(self.twtt, time)
                        self.amp = np.append(self.amp, coeff)
                        return
                R = ((self.impedance[index-1] - self.impedance[index]) / (self.impedance[index-1] + self.impedance[index])) * coeff
                T = ((2*self.impedance[index-1]) / (self.impedance[index-1] + self.impedance[index])) * coeff
                if T >= self.thresh:
                    self.pulse(index-1, 1, time, T)
                if R >= self.thresh:
                    self.pulse(index, -1, time, R)


def run():
    layers = [10, 15, 20]
    rho = [1026, 1906, 1922, 1955]
    c = [1500, 1600, 1650, 1660]
    thresh = 0.000001
    m_per_p = 1 # Meters/Pixel
    sbp_simulate = SBP_Simulate(layers, rho, c, m_per_p, thresh)
    sbp_simulate.simulate() 
    print(sbp_simulate.twtt)

    plt.plot(sbp_simulate.twtt, 20*np.log(sbp_simulate.amp), "o")
    plt.show()


if __name__ == '__main__':
    run()