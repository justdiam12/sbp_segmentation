import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import io
import scipy.io as sio
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import find_peaks

# Ricker Wavelet
# Question about spherical spreading and attenuation. Add attenuation tomorrow when clarified.

class SBP_Simulate_v2():
    def __init__(self, time, layers, rho, c, attenuation, sample_f, time_interval):
        self.time = time # This is the time the sonar system travels over an area of interest (s)
        self.rho = rho # Density of each layer (kg/m^3)
        self.c = c # Sound Speed in each layer (m/s)
        self.attenuation = attenuation # Attenuation in each layer (dB/m)
        self.sample_f = sample_f # Sonar sampling frequency (Hz)
        self.time_interval = time_interval # Frequency of the Sonar System (Hz)
        self.pulse_width = 0.3 # Time between sonar pulses (s)
        # self.size = int(self.time * self.sample_f) # Length of the trackline 
        self.size = 1000
        self.layers = random_contour(layers, self.size) 

        # Seafloor parameter profiles
        self.rho_profile = np.zeros((len(self.rho), self.size))
        self.c_profile = np.zeros((len(self.c), self.size))
        self.attenuation_profile = np.zeros((len(self.attenuation), self.size))
        for j in range(self.size):
            self.rho_profile[:,j] = self.rho
            self.c_profile[:,j] = self.c
            self.attenuation_profile[:,j] = self.rho
        self.impedance = self.rho_profile * self.c_profile

        # TWTT and Amplitude arrays
        self.twtt = np.arange(0, self.pulse_width, self.time_interval, dtype=np.float32)
        self.amp = np.ones((len(self.twtt), self.size), dtype=np.float32)*0.0000001

    def simulate(self):
        layer = 0
        up_down = -1 
        t_elapsed = 0
        t_in_layer = 0
        dist_traveled = 0
        dist_in_layer = 0
        coeff = 1
        rt = []
        for j in range(self.size):
            self.pulse(layer, j, up_down, t_elapsed, t_in_layer, dist_traveled, dist_in_layer, coeff, rt)            

    def pulse(self, layer, dim, up_down, time_elapsed, t_in_layer, dist_traveled, dist_in_layer, coeff, rt):
        if up_down == -1: # Descending Pulse
            if time_elapsed > self.pulse_width or np.abs(coeff) < 0.000001:
                return
            if layer + 1 >= self.layers.shape[0]:
                return
            else:
                while dist_in_layer < self.layers[layer, dim] and time_elapsed < self.pulse_width:
                    time_elapsed += self.time_interval
                    t_in_layer += self.time_interval
                    dist_traveled += self.c_profile[layer, dim] * self.time_interval
                    dist_in_layer += self.c_profile[layer, dim] * self.time_interval
                
                # Fix for overshoot in distance
                if layer == 0:
                    dist_difference = self.layers[layer, dim] - dist_in_layer
                else:
                    dist_difference = self.layers[layer+1, dim] - self.layers[layer, dim] - dist_in_layer
                dist_in_layer += dist_difference
                time_difference = dist_difference / self.c_profile[layer, dim]
                t_in_layer += time_difference
                
                # Compute R and T coefficients and recursive pulse call
                R = ((self.impedance[layer+1, dim] - self.impedance[layer, dim]) / (self.impedance[layer+1, dim] + self.impedance[layer, dim])) * np.exp(-self.attenuation[layer]*dist_in_layer) * coeff
                T = ((2*self.impedance[layer+1, dim]) / (self.impedance[layer+1, dim] + self.impedance[layer, dim])) * np.exp(-self.attenuation[layer]*dist_in_layer) * coeff
                self.pulse(layer-1, dim, 1, time_elapsed + time_difference, 0, dist_traveled + dist_difference, 0, R, np.append(rt,R))
                if layer+1 < self.layers.shape[0]:
                    self.pulse(layer+1, dim, -1, time_elapsed + time_difference, 0, dist_traveled + dist_difference, 0, T, np.append(rt,T))

        elif up_down == 1: # Ascending Pulse
            if time_elapsed > self.pulse_width or np.abs(coeff) < 0.000001:
                return
            if layer == -1:
                layer = 0
                while dist_in_layer < self.layers[layer, dim] and time_elapsed < self.pulse_width:
                    time_elapsed += self.time_interval
                    t_in_layer += self.time_interval
                    dist_traveled += self.c_profile[layer, dim] * self.time_interval
                    dist_in_layer += self.c_profile[layer, dim] * self.time_interval
                    
                # Fix for overshoot in distance
                dist_difference = self.layers[layer, dim] - dist_in_layer
                dist_traveled += dist_difference
                dist_in_layer += dist_difference
                time_difference = dist_difference / self.c_profile[layer, dim]
                time_elapsed += time_difference
                t_in_layer += time_difference

                # Append the amplitude information
                index = np.where(np.isclose(self.twtt, time_elapsed, atol = self.time_interval))[0]
                self.convolve(index, dim, coeff, layer, dist_in_layer, dist_traveled)
                self.disp_info(time_elapsed, index, coeff, dist_traveled)

            else:
                while dist_in_layer < self.layers[layer, dim] and time_elapsed < self.pulse_width:
                    time_elapsed += self.time_interval
                    t_in_layer += self.time_interval
                    dist_traveled += self.c_profile[layer, dim] * self.time_interval
                    dist_in_layer += self.c_profile[layer, dim] * self.time_interval

                # Fix for overshoot in distance
                dist_difference = self.layers[layer+1, dim] - self.layers[layer, dim] - dist_in_layer
                dist_in_layer += dist_difference
                time_difference = dist_difference / self.c_profile[layer, dim]
                t_in_layer += time_difference

                # Compute R and T coefficients and recursive pulse call
                R = ((self.impedance[layer, dim] - self.impedance[layer+1, dim]) / (self.impedance[layer, dim] + self.impedance[layer+1, dim])) * np.exp(-self.attenuation[layer+1]*dist_in_layer) * coeff
                T = ((2*self.impedance[layer, dim]) / (self.impedance[layer, dim] + self.impedance[layer+1, dim])) * np.exp(-self.attenuation[layer+1]*dist_in_layer) * coeff
                self.pulse(layer+1, dim, -1, time_elapsed + time_difference, 0, dist_traveled + dist_difference, 0, R, np.append(rt,R))
                if layer+1 < self.layers.shape[0]:
                    self.pulse(layer-1, dim, 1, time_elapsed + time_difference, 0, dist_traveled + dist_difference, 0, T, np.append(rt,T))

    def convolve(self, index, dim, coeff, layer, dist_in_layer, dist_traveled):
        

        self.amp[index, dim] = coeff * np.exp(-self.attenuation[layer]*dist_in_layer) / dist_traveled

    def inversion(self):
        return

    def disp_info(self, time_elapsed, index, coeff, dist_traveled):
        print("Time Elapsed: ", time_elapsed)
        print("TWTT: ", self.twtt[index])
        print("Coeff: ", coeff)
        print("Distance Traveled: ", dist_traveled)
        print(" ")


def random_contour(layers, size):
    layer_map = np.zeros((len(layers), size))

    for i in range(layer_map.shape[0]):
        for j in range(layer_map.shape[1]):
            layer_map[i,j] = layers[i]
    return layer_map


def run():
    time = 0.1
    layers = [50, 100, 150]
    rho = [1026, 1600, 1800, 2000]
    c = [1500, 1450, 1700, 1900]
    attenuation = [0, 0, 0, 0] # Write a function that determines attenuation based off of system frequency and properties
    sample_f = 16.67e3
    time_interval = 0.0001
    sbp_simulate = SBP_Simulate_v2(time, layers, rho, c, attenuation, sample_f, time_interval)
    sbp_simulate.simulate()
    sbp_simulate.inversion()

    # Plots 
    plt.figure(1)
    plt.imshow(np.abs(sbp_simulate.amp), cmap='gray')
    plt.savefig('amp.png')
    plt.show()


if __name__ == '__main__':
    run()