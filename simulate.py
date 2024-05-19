# This is the code to simulate the SBP data
import numpy as np
import matplotlib.pyplot as plt

class SBP_Simulate():
    def __init__(self, size, layers, rho, c, m_per_p, thresh, frequency):
        self.size = size
        self.layers = layers
        self.rho = rho
        self.c = c
        self.m_per_p = m_per_p
        self.thresh = thresh
        self.frequency = frequency
        self.profile = np.ones((self.size,1), dtype=np.float32)*0.00001
        self.floor = np.ones((self.size, self.size), dtype=np.float32)*0.00001
        self.rho_profile = np.zeros((self.size, 1))
        self.c_profile = np.zeros((self.size, 1))
        self.twtt = np.arange(0, 1, 1/self.frequency, dtype=np.float32)
        self.amp = np.zeros_like(self.twtt)*0.000001
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
        rt = []
        self.pulse(index, up_down, time, coeff, rt)

    def pulse(self, index, up_down, time, coeff, rt):
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
                if np.abs(R) >= self.thresh:
                    self.pulse(index, 1, time + self.m_per_p / self.c_profile[index], R, np.append(rt,R))
                if np.abs(T) >= self.thresh:
                    self.pulse(index+1, -1, time + self.m_per_p / self.c_profile[index], T, np.append(rt,T))

        elif up_down == 1: # Ascending Pulse
            if index-1 == -1:
                return
            else:
                while self.impedance[index] == self.impedance[index-1]:
                    time += self.m_per_p / self.c_profile[index]
                    index -= 1
                    if index-1 == -1:
                        time += self.m_per_p / self.c_profile[index]
                        print(np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0])
                        self.amp[np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0]] = coeff
                        # print(rt)
                        return
                R = ((self.impedance[index-1] - self.impedance[index]) / (self.impedance[index-1] + self.impedance[index])) * coeff
                T = ((2*self.impedance[index-1]) / (self.impedance[index-1] + self.impedance[index])) * coeff
                if np.abs(T) >= self.thresh:
                    self.pulse(index-1, 1, time + self.m_per_p / self.c_profile[index], T, np.append(rt,T))
                if np.abs(R) >= self.thresh:
                    self.pulse(index, -1, time + self.m_per_p / self.c_profile[index], R, np.append(rt,R))

    def conv(self):
        exp_decay = np.exp(-np.arange(0, 0.05, 1 / self.frequency, dtype=np.float32) * 200)
        max_amp_before = np.max(np.abs(self.amp)) 
        convolved_amp = np.convolve(self.amp, exp_decay, 'same') 
        max_amp_after = np.max(np.abs(convolved_amp))
        if max_amp_after != 0:
            self.amp = convolved_amp * (max_amp_before / max_amp_after)
        else:
            self.amp = convolved_amp

    def seafloor(self):
        for i in range(self.size):
            segment_size = len(self.amp) // self.size
            self.floor[:,i] = np.mean(self.amp[:segment_size*self.size].reshape(-1, segment_size), axis=1)

def run():
    size = 2500
    # layers = [10, 20, 30]
    layers = [50, 115, 200]
    rho = [1026, 1600, 1800, 2000]
    c = [1500, 1450, 1700, 1900]
    thresh = 0.001
    m_per_p = 1 # Meters/Pixel
    frequency = 16.67e3
    sbp_simulate = SBP_Simulate(size, layers, rho, c, m_per_p, thresh, frequency)
    sbp_simulate.simulate() 
    sbp_simulate.conv() 
    sbp_simulate.seafloor()

    plt.figure(1)
    plt.plot(sbp_simulate.twtt, np.abs(sbp_simulate.amp), "-r")

    plt.figure(2)
    plt.imshow(np.abs(sbp_simulate.floor), extent=[0, sbp_simulate.floor.shape[1], sbp_simulate.twtt[-1], sbp_simulate.twtt[0]], cmap='gray', aspect='auto')
    plt.colorbar(label="Amplitude")  
    plt.title('Seafloor Simulation')
    plt.xlabel('Distance (m)')
    plt.ylabel('TWTT')
    plt.yticks(np.arange(sbp_simulate.twtt[0], sbp_simulate.twtt[-1], 0.1))
    plt.show()


if __name__ == '__main__':
    run()