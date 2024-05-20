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
        self.floor = np.ones((self.size, self.size), dtype=np.float32)*0.000001
        self.rho_profile = np.zeros((self.size, self.size))
        self.c_profile = np.zeros((self.size, self.size))
        self.twtt = np.arange(0, 1, 1/self.frequency, dtype=np.float32)
        self.amp = np.ones((len(self.twtt), size), dtype=np.float32)*0.000001
        for j in range(self.layers.shape[1]):
            for i in range(self.layers.shape[0]):
                layer_1 = int(self.layers[i,j])
                if i == 0:
                    layer_2 = int(self.layers[i+1,j])
                    self.rho_profile[0:layer_1, j] = self.rho[i]
                    self.c_profile[0:layer_1, j] = self.c[i]
                    self.rho_profile[layer_1:layer_2, j] = self.rho[i+1]
                    self.c_profile[layer_1:layer_2, j] = self.c[i+1]
                elif i == self.layers.shape[0]-1:
                    self.rho_profile[layer_1:, j] = self.rho[i+1]
                    self.c_profile[layer_1:, j] = self.c[i+1]
                else:
                    layer_2 = int(self.layers[i+1,j])
                    self.rho_profile[layer_1:layer_2, j] = self.rho[i+1]
                    self.c_profile[layer_1:layer_2,j] = self.c[i+1]
        self.impedance = self.rho_profile * self.c_profile
        
    def simulate(self):
        index = 0
        up_down = -1 
        time = 0
        coeff = 1
        rt = []
        for i in range(self.size):
            self.pulse(index, i, up_down, time, coeff, rt)

    def pulse(self, index, dim, up_down, time, coeff, rt):
        if up_down == -1: # Descending Pulse
            if index+1 == self.size:
                return
            else:
                while self.impedance[index, dim] == self.impedance[index+1, dim]:
                    time += self.m_per_p / self.c_profile[index, dim]
                    index += 1
                    if index+1 == self.size:
                        return
                R = ((self.impedance[index+1, dim] - self.impedance[index, dim]) / (self.impedance[index+1, dim] + self.impedance[index, dim])) * coeff
                T = ((2*self.impedance[index+1, dim]) / (self.impedance[index+1, dim] + self.impedance[index, dim])) * coeff
                if np.abs(R) >= self.thresh:
                    self.pulse(index, dim, 1, time + self.m_per_p / self.c_profile[index, dim], R, np.append(rt,R))
                if np.abs(T) >= self.thresh:
                    self.pulse(index+1, dim, -1, time + self.m_per_p / self.c_profile[index, dim], T, np.append(rt,T))

        elif up_down == 1: # Ascending Pulse
            if index-1 == -1:
                return
            else:
                while self.impedance[index, dim] == self.impedance[index-1, dim]:
                    time += self.m_per_p / self.c_profile[index, dim]
                    index -= 1
                    if index-1 == -1:
                        time += self.m_per_p / self.c_profile[index, dim]
                        self.amp[np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0], dim] = coeff
                        # print(rt)
                        return
                R = ((self.impedance[index-1, dim] - self.impedance[index, dim]) / (self.impedance[index-1, dim] + self.impedance[index, dim])) * coeff
                T = ((2*self.impedance[index-1, dim]) / (self.impedance[index-1, dim] + self.impedance[index, dim])) * coeff
                if np.abs(T) >= self.thresh:
                    self.pulse(index-1, dim, 1, time + self.m_per_p / self.c_profile[index, dim], T, np.append(rt,T))
                if np.abs(R) >= self.thresh:
                    self.pulse(index, dim, -1, time + self.m_per_p / self.c_profile[index, dim], R, np.append(rt,R))

    def conv(self):
        for i in range(self.size):
            exp_decay = np.exp(-np.arange(0, 0.05, 1 / self.frequency, dtype=np.float32) * 200)
            max_amp_before = np.max(np.abs(self.amp[:,i])) 
            convolved_amp = np.convolve(self.amp[:,i], exp_decay, 'same') 
            max_amp_after = np.max(np.abs(convolved_amp))
            if max_amp_after != 0:
                self.amp[:,i] = convolved_amp * (max_amp_before / max_amp_after)
            else:
                self.amp[:,i] = convolved_amp


def random_contour(size, layers):
    layer_map = np.zeros((len(layers), size))
    return layer_map

def run():
    size = 2500
    layers = [50, 115, 200]
    # for i in range(size):
    #     layer_map[:,i] = layers
    layer_map = random_contour(size, layers)
    rho = [1026, 1600, 1800, 2000]
    c = [1500, 1450, 1700, 1900]
    thresh = 0.001
    m_per_p = 1 # Meters/Pixel
    frequency = 16.67e3
    sbp_simulate = SBP_Simulate(size, layer_map, rho, c, m_per_p, thresh, frequency)
    sbp_simulate.simulate() 
    sbp_simulate.conv() 

    plt.figure(1)
    plt.plot(sbp_simulate.twtt, np.abs(sbp_simulate.amp[:,0]), "-r")

    plt.figure(2)
    plt.imshow(np.abs(sbp_simulate.amp), extent=[0, sbp_simulate.floor.shape[1], sbp_simulate.twtt[-1], sbp_simulate.twtt[0]], cmap='gray', aspect='auto')
    plt.colorbar(label="Amplitude")  
    plt.title('Seafloor Simulation')
    plt.xlabel('Distance (m)')
    plt.ylabel('TWTT')
    plt.yticks(np.arange(sbp_simulate.twtt[0], sbp_simulate.twtt[-1], 0.1))
    plt.show()


if __name__ == '__main__':
    run()