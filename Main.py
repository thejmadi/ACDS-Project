# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:16:35 2023

@author: tarun
"""
import Functions as fc
import matplotlib.pyplot as plt
import numpy as np

def main():
    const = dict(a = 7000, i = np.radians(83), Om = np.radians(-70), f = np.radians(0), arg_per = np.radians(0),
                 D = 3, I_sat = np.diag([4500, 6000, 7000]), I_w = 0.5, plate_areas = np.array([5, 7, 7]),
                 plate_CoM = np.array([[0.1, 0.1, 0], [2, 0, 0], [-2, 0, 0]]), plate_abs = np.array([0.1, 0.7, 0.7]),
                 plate_spec = np.array([0.8, 0.2, 0.2]), plate_diff = np.array([0.1, 0.1, 0.1]),
                 mu = 3.986004354360959e5, w0 = 7.29212e-5)
    Period = 2*np.pi*np.sqrt((const['a']**3)/const["mu"])
    t = np.arange(0, 3*Period, 0.1)
    
    angles = np.zeros((6, t.size))
    angles[:, 0] = np.radians(np.array([5, -4, 10, 0.5, 0.7, 6]))
    
    fc.RK4(t, angles, const)
    plt.plot(t, angles[0])
    plt.show()
    plt.plot(t, angles[1])
    plt.show()
    plt.plot(t, angles[2])
    plt.show()
    plt.plot(t, angles[3])
    plt.show()
    plt.plot(t, angles[4])
    plt.show()
    plt.plot(t, angles[5])

main()