import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import io
import scipy.io as sio
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import find_peaks

class SBP_Simulate_v2():
    def __init__(self, size, rho, c, threshold, frequency):
        self.size = size
        self.rho = rho
        self.c = c
        self.threshold = threshold
        self.frequency = frequency
        self.

        def simulate(self):
            index = 0
            up_down = -1 
            time = 0
            coeff = 1000
            rt = []
            _range = 0
            for j in range(self.size):
                self.pulse(index, _range, j, up_down, time, coeff, rt)
                sorted_data = sorted(self.amp_indices, key=lambda x: x[1], reverse=True)
                for i in range(self.layers.shape[0]+1):
                    data = sorted_data[i]
                    if i == 0:
                        data_2 = sorted_data[i+1]
                        self.mask[0:data[0], j] = i
                        self.mask[data[0]:data_2[0], j] = i+1
                    elif i == self.layers.shape[0]:
                        self.mask[data[0]:, j] = i
                    else:
                        data_2 = sorted_data[i+1]
                        self.mask[data[0]:data_2[0], j] = i+1
                        
                self.amp_indices = []

        def pulse(self, index, _range, dim, up_down, time, coeff, rt):
            if up_down == -1: # Descending Pulse
                if index+1 == self.size:
                    return
                else:
                    while self.impedance[index, dim] == self.impedance[index+1, dim]:
                        time += self.m_per_p / self.c_profile[index, dim]
                        index += 1
                        _range += 1
                        if index+1 == self.size:
                            return
                    R = ((self.impedance[index+1, dim] - self.impedance[index, dim]) / (self.impedance[index+1, dim] + self.impedance[index, dim])) * coeff
                    T = ((2*self.impedance[index+1, dim]) / (self.impedance[index+1, dim] + self.impedance[index, dim])) * coeff
                    if np.abs(R) >= self.thresh:
                        self.pulse(index, _range + 1, dim, 1, time + self.m_per_p / self.c_profile[index, dim], R, np.append(rt,R))
                    if np.abs(T) >= self.thresh:
                        self.pulse(index+1, _range + 1, dim, -1, time + self.m_per_p / self.c_profile[index, dim], T, np.append(rt,T))

            elif up_down == 1: # Ascending Pulse
                if index-1 == -1:
                    return
                else:
                    while self.impedance[index, dim] == self.impedance[index-1, dim]:
                        time += self.m_per_p / self.c_profile[index, dim]
                        index -= 1
                        _range += 1
                        if index-1 == -1:
                            time += self.m_per_p / self.c_profile[index, dim]
                            _range += 1
                            if len(np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0]) > 0:
                                dimes = np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0]
                                self.amp[dimes[0], dim] = coeff/_range
                                self.amp_indices.append((int(np.round((dimes[0]/self.twtt.shape[0])*(self.size-1))), coeff/_range))

                            # print(20*np.log(rt))
                            # print(time)
                            return
                    R = ((self.impedance[index-1, dim] - self.impedance[index, dim]) / (self.impedance[index-1, dim] + self.impedance[index, dim])) * coeff
                    T = ((2*self.impedance[index-1, dim]) / (self.impedance[index-1, dim] + self.impedance[index, dim])) * coeff
                    if np.abs(T) >= self.thresh:
                        self.pulse(index-1, _range + 1, dim, 1, time + self.m_per_p / self.c_profile[index, dim], T, np.append(rt,T))
                    if np.abs(R) >= self.thresh:
                        self.pulse(index, _range + 1, dim, -1, time + self.m_per_p / self.c_profile[index, dim], R, np.append(rt,R))


def run():

    sbp_simulate = SBP_Simulate_v2()

if __name__ == '__main__':
    run()

