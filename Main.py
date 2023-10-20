# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:16:35 2023

@author: tarun
"""
import Functions as fc
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate as sci

def main():
    const = dict(a = 7000, i = np.radians(83), Om = np.radians(-70), f = np.radians(0), arg_per = np.radians(0),
                 D = 3, I_sat = np.diag([4500, 6000, 7000]), I_w = 0.5, plate_areas = np.array([5, 7, 7]),
                 plate_CoM = np.array([[0.1, 0.1, 0], [2, 0, 0], [-2, 0, 0]]), plate_abs = np.array([0.1, 0.7, 0.7]),
                 plate_spec = np.array([0.8, 0.2, 0.2]), plate_diff = np.array([0.1, 0.1, 0.1]),
                 mu = 3.986004354360959e5)
    #const["w0"] = 7.29212e-5
    const["w0"] = np.sqrt(const["mu"]/const["a"]**3)
    Period = 2*np.pi*np.sqrt((const['a']**3)/const["mu"])
    t = np.arange(0, 3*Period, 1)
    
    x = np.zeros((6, t.size))
    x[:, 0] = np.array([-0.07, 0.09,  0.15, -8e-4, 6e-4, 8.5e-4])
    
    x = fc.RK4(t, x, const)
    #res = sci.solve_ivp(fc.scipy, [0, 3*Period], x[:, 0], t_eval=t)
    
    plt.plot(t, x[0])
    plt.title("Roll vs. Time")
    plt.show()
    plt.plot(t, x[1])
    plt.title("Yaw vs. Time")
    plt.show()
    plt.plot(t, x[2])
    plt.title("Pitch vs. Time")
    plt.show()
    
    '''
    plt.plot(t, x[3])
    plt.show()
    plt.plot(t, x[4])
    plt.show()
    plt.plot(t, x[5])
    plt.show()
    '''
    '''
    plt.plot(res.t, res.y[0])
    plt.title("Roll vs. Time")
    plt.show()
    plt.plot(res.t, res.y[1])
    plt.title("Yaw vs. Time")
    plt.show()
    plt.plot(res.t, res.y[2])
    plt.title("Pitch vs. Time")
    '''
   # plt.show()
    '''
    plt.plot(res.t, res.y[3])
    plt.show()
    plt.plot(res.t, res.y[4])
    plt.show()
    plt.plot(res.t, res.y[5])
    plt.show()
    '''
    

main()