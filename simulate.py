# This is the code to simulate the SBP data
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import io
from scipy.ndimage import uniform_filter

class SBP_Simulate():
    def __init__(self, size, layers, rho, c, m_per_p, thresh, frequency):
        self.size = size
        self.mask = np.zeros((self.size, self.size))
        self.layers = layers
        self.rho = rho
        self.c = c
        self.m_per_p = m_per_p
        self.thresh = thresh
        self.frequency = frequency
        self.floor = np.ones((self.size, self.size), dtype=np.float32)*0.00000001
        self.rho_profile = np.zeros((self.size, self.size))
        self.c_profile = np.zeros((self.size, self.size))
        self.twtt = np.arange(0, 0.4, 1/self.frequency, dtype=np.float32)
        self.amp = np.ones((len(self.twtt), size), dtype=np.float32)*0.0000001
        self.amp_indices = []
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
        coeff = 1000
        rt = []
        _range = 0
        for j in range(self.size):
            self.pulse(index, _range, j, up_down, time, coeff, rt)
            sorted_data = sorted(self.amp_indices, key=lambda x: np.abs(x[1]), reverse=True)
            for i in range(self.layers.shape[0]):
                data = sorted_data[i]
                if i == 0:
                    data_2 = sorted_data[i+1]
                    self.mask[0:data[0], j] = i
                    self.mask[data[0]:data_2[0], j] = i+1
                elif i == self.layers.shape[0]-1:
                    self.mask[data[0]:, j] = i+1
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
                        self.amp[np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0], dim] = coeff/_range
                        if len(np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0]) > 0:
                            dimes = np.where(np.isclose(self.twtt, time, atol=1/self.frequency))[0]
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

    def conv(self):
        for i in range(self.size):
            exp_decay = np.exp(-np.arange(0, 0.05, 1 / self.frequency, dtype=np.float32) * 75)
            max_amp_before = np.max(np.abs(self.amp[:,i])) 
            convolved_amp = uniform_filter(self.amp[:,i], size=10, mode='reflect')
            convolved_amp = uniform_filter(np.convolve(self.amp[:,i], exp_decay, 'same'), size=10, mode='reflect')
            max_amp_after = np.max(np.abs(convolved_amp))
            if max_amp_after != 0:
                self.amp[:,i] = convolved_amp * (max_amp_before / max_amp_after)
            else:
                self.amp[:,i] = convolved_amp
            # Add Gaussian noise
            noise_mean = 0  # Mean of the Gaussian noise
            noise_std = 0.1 * max_amp_before  # Standard deviation of the noise
            gaussian_noise = np.random.normal(noise_mean, noise_std, self.amp[:, i].shape)
            self.amp[:, i] += gaussian_noise

    def create_mask(self):
        for j in range(self.mask.shape[0]):
            index = 0
            mask_layer = 0
            for i in range(self.mask.shape[1]):
                if self.mask[i,j] == 100:
                    self.mask[index:i+1,j] = mask_layer
                    mask_layer += 1
                    index = i+1


def random_contour(size, layers):
    layer_map = np.zeros((len(layers), size))

    for i in range(layer_map.shape[0]):
        for j in range(layer_map.shape[1]):
            if j == 0:
                layer_map[i,j] = layers[i]
            else:
                layer_map[i,j] = random.randint(layers[i-1]-2, layers[i-1]+2)

    return layer_map

def nemp_contours(filename):
    matfile = io.loadmat(filename)
    label = matfile["label"]
    map = np.zeros((label.shape[0], label.shape[1]))
    nemp_map = np.zeros((label.shape[2]-1, label.shape[1]))
    for i in range(label.shape[2]):
        for x in range(label.shape[0]):
            for y in range(label.shape[1]):
                if label[x, y, i] == 1:
                    map[x,y] = int(i)
    
    for j in range(map.shape[1]):
        index = 0
        for i in range(map.shape[0]):
            if i+1 != map.shape[0]:
                if map[i+1,j] != map[i,j]:
                    nemp_map[index, j] = i
                    index += 1

    return nemp_map

def run():
    size = 256
    # layers = [50, 70, 100, 120, 200]
    # layer_map = random_contour(size, layers)
    filename = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Train/nemp_data_9451"
    layer_map = nemp_contours(filename)
    # layer_map = random_contour(size, layers)
    rho = [1026, 1600, 1800, 2000, 2200, 2400]
    c = [1500, 1450, 1700, 1900, 2000, 2200]
    thresh = 0.01
    m_per_p = 1 # Meters/Pixel
    frequency = 16.67e3
    sbp_simulate = SBP_Simulate(size, layer_map, rho, c, m_per_p, thresh, frequency)
    sbp_simulate.simulate() 
    sbp_simulate.conv() 
    sbp_simulate.create_mask()

    plt.figure(1)
    plt.plot(sbp_simulate.twtt, np.abs(sbp_simulate.amp[:,0]), "-r")
    plt.savefig("amp.png")

    plt.figure(2)
    plt.imshow(20*np.log(np.abs(sbp_simulate.amp)), extent=[0, sbp_simulate.floor.shape[1], sbp_simulate.twtt[-1], sbp_simulate.twtt[0]], cmap='gray', aspect ='auto')
    plt.colorbar(label="Amplitude")  
    plt.title('Seafloor Simulation')
    plt.xlabel('Distance (m)')
    plt.ylabel('TWTT')
    plt.clim(-50, 0)
    plt.yticks(np.arange(sbp_simulate.twtt[0], sbp_simulate.twtt[-1], 0.1))
    plt.savefig("contour.png")

    plt.figure(3)
    plt.imshow(sbp_simulate.mask)
    plt.title("Simulate Mask")
    plt.show()


if __name__ == '__main__':
    run()