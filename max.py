import matplotlib.pyplot as plt 
import numpy as np
from simulate import SBP_Simulate
from simulate import nemp_contours
from simulate import random_contour

def find_max(amp):
    max_amp = np.array([], dtype=np.float32)

    for i in range(len(amp)):
        if i == 0:
            if amp[i] > amp[i+1]:
                max_amp = np.append(max_amp, (i, amp[i]/1000))
        elif i == len(amp)-1:
            if amp[i] > amp[i-1]:
                max_amp = np.append(max_amp, (i, amp[i]/1000))
        else:
            if amp[i] > amp[i-1] and amp[i] > amp[i+1]:
                max_amp = np.append(max_amp, (i, amp[i]/1000))

    max_amp = np.polyfit(np.arange(0, len(max_amp)), max_amp, 10)


def run():
    size = 256
    # layers = [50, 70, 100, 120, 200]
    # layer_map = random_contour(size, layers)
    filename = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Train/nemp_data_9451"
    layer_map = nemp_contours(filename)
    # layer_map = random_contour(size, layers)
    rho = [1026, 1600, 1800, 2000, 2200, 2400]
    c = [1500, 1450, 1700, 1900, 2000, 2200]
    thresh = 0.1
    m_per_p = 1 # Meters/Pixel
    frequency = 16.67e3
    sbp_simulate = SBP_Simulate(size, layer_map, rho, c, m_per_p, thresh, frequency)
    sbp_simulate.simulate() 
    sbp_simulate.conv()
    # sbp_simulate.create_mask()

    # plt.figure(1)
    # plt.plot(sbp_simulate.twtt, np.abs(sbp_simulate.amp[:,0]), "-r")

    max_amp = find_max(sbp_simulate.amp[:,0])

    plt.plot(range(len(max_amp)), max_amp)
    plt.show()


if __name__ == '__main__':
    run()