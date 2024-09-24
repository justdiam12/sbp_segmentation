import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import io
import scipy.io as sio
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import find_peaks

# Question about spherical spreading and attenuation. Add attenuation tomorrow when clarified.

class SBP_Simulate_v2():
    def __init__(self, time, layers, rho, c, attenuation, sample_f, time_interval):
        self.time = time # This is the time the sonar system travels over an area of interest (s)
        self.rho = rho # Density of each layer (kg/m^3)
        self.c = c # Sound Speed in each layer (m/s)
        self.attenuation = attenuation # Attenuation in each layer (dB/m)
        self.sample_f = sample_f # Sonar sampling frequency (Hz)
        self.time_interval = time_interval # Frequency of the Sonar System (Hz)
        self.pulse_width = 1 / self.sample_f # Time between sonar pulses (s)
        self.size = self.time * self.sample_f # Length of the trackline 
        self.layers = random_contour(layers, self.size) 

        # Seafloor parameter profiles
        self.rho_profile = np.zeros((len(self.rho), self.size))
        self.c_profile = np.zeros((len(self.c), self.size))
        self.attenuation_profile = np.zeros((len(self.attenuation), self.size))
        for j in range(self.size):
            for i in range(self.layers.shape[0]):
                layer_1 = int(self.layers[i,j])
                if i == 0:
                    layer_2 = int(self.layers[i+1,j])
                    self.rho_profile[0:layer_1, j] = self.rho[i]
                    self.c_profile[0:layer_1, j] = self.c[i]
                    self.attenuation_profile[0:layer_1, j] = self.attenuation[i]
                    self.rho_profile[layer_1:layer_2, j] = self.rho[i+1]
                    self.c_profile[layer_1:layer_2, j] = self.c[i+1]
                    self.attenuation_profile[layer_1:layer_2, j] = self.attenuation[i+1]
                elif i == self.layers.shape[0]-1:
                    self.rho_profile[layer_1:, j] = self.rho[i+1]
                    self.c_profile[layer_1:, j] = self.c[i+1]
                    self.attenuation_profile[layer_1:, j] = self.attenuation[i+1]
                else:
                    layer_2 = int(self.layers[i+1,j])
                    self.rho_profile[layer_1:layer_2, j] = self.rho[i+1]
                    self.c_profile[layer_1:layer_2,j] = self.c[i+1]
                    self.attenuation_profile[layer_1:layer_2,j] = self.attenuation[i+1]
        self.impedance = self.rho_profile * self.c_profile

        # TWTT and Amplitude arrays
        self.twtt = np.arange(0, self.pulse_width, self.time_interval, dtype=np.float32)
        self.amp = np.ones((len(self.twtt), self.size), dtype=np.float32)*0.0000001

    def simulate(self):
        layer = 0
        up_down = -1 
        t_0 = 0
        t_layer = 0
        dist = 0
        coeff = 1
        rt = []
        for j in range(self.size):
            self.pulse(layer, j, up_down, t_0, t_layer, dist, coeff, rt)
            

    def pulse(self, layer, dim, up_down, time, t_layer, dist, coeff, rt):
        if up_down == -1: # Descending Pulse
            if time + self.time_interval >= self.pulse_width:
                return
            else:
                while dist < self.layers[layer, dim]:
                    time += self.time_interval
                    t_layer += self.time_interval
                    dist += self.c_profile[layer, dim] * self.time_interval
                    if time + self.time_interval >= self.pulse_width:
                        return
                R = ((self.impedance[layer+1, dim] - self.impedance[layer, dim]) / (self.impedance[layer+1, dim] + self.impedance[layer, dim]))*coeff
                T = ((2*self.impedance[layer+1, dim]) / (self.impedance[layer+1, dim] + self.impedance[layer, dim]))*coeff
                self.pulse(layer+1, dist + 1, dim, 1, time + self.m_per_p / self.c_profile[index, dim], R, np.append(rt,R))
                if layer+1 < len(self.layers.shape[0]):
                    self.pulse(layer+1, dist, dim, -1, time, T, np.append(rt,T))

        elif up_down == 1: # Ascending Pulse
            if layer == 0:
                while t_layer < self.layers[layer, dim]/self.c_profile[layer, dim] and time < self.pulse_width:
                    time += self.time_interval
                    t_layer += self.time_interval
                    dist += self.c_profile[layer, dim] * self.time_interval
                    if time + self.time_interval >= self.pulse_width:
                        return
                index = np.where(self.twtt == time)
                self.amp[index, dim] = coeff
            else:
                while self.impedance[index, dim] == self.impedance[index-1, dim]:
                    time += self.m_per_p / self.c_profile[index, dim]
                    index -= 1
                    dist += 1
                    if index-1 == -1:
                        time += self.m_per_p / self.c_profile[index, dim]
                        dist += 1
                        if len(np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0]) > 0:
                            inde = np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0]
                            self.amp[dimes[0], dim] = coeff/dist
                            self.amp_indices.append((int(np.round((dimes[0]/self.twtt.shape[0])*(self.size-1))), coeff/dist))
                        return
                R = ((self.impedance[index-1, dim] - self.impedance[index, dim]) / (self.impedance[index-1, dim] + self.impedance[index, dim])) * coeff
                T = ((2*self.impedance[index-1, dim]) / (self.impedance[index-1, dim] + self.impedance[index, dim])) * coeff
                if np.abs(T) >= self.thresh:
                    self.pulse(index-1, dist + 1, dim, 1, time + self.m_per_p / self.c_profile[index, dim], T, np.append(rt,T))
                if np.abs(R) >= self.thresh:
                    self.pulse(index, dist + 1, dim, -1, time + self.m_per_p / self.c_profile[index, dim], R, np.append(rt,R))


def random_contour(layers, size):
    layer_map = np.zeros((len(layers), size))

    for i in range(layer_map.shape[0]):
        for j in range(layer_map.shape[1]):
            layer_map[i,j] = layers[i]
    return layer_map


def run():
    time = 5
    layers = [50, 100, 150]
    rho = [1026, 1600, 1800, 2000]
    c = [1500, 1450, 1700, 1900]
    attenuation = [0, 0, 0, 0]
    sample_f = 16.67e3
    time_interval = 0.0001
    sbp_simulate = SBP_Simulate_v2(layers, rho, c, attenuation, sample_f, time_interval)

    # Plots 
    plt.figure(1)
    plt.plot(sbp_simulate.twtt, np.abs(sbp_simulate.amp[:,0]), '-r')
    plt.show()

if __name__ == '__main__':
    run()