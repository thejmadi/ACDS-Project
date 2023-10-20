# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:09:23 2023

@author: tarun
"""

import numpy as np

def xDot(t, x, const, axis):
    phi = x[0]
    if np.cos(phi) == 0:
        phi += 1e-10
    psi = x[1]
    if np.cos(psi) == 0:
        psi += 1e-10
    th = x[2]
    w1 = x[3]
    w2 = x[4]
    w3 = x[5]
    I1 = const["I_sat"][0, 0]
    I2 = const["I_sat"][1, 1]
    I3 = const["I_sat"][2, 2]
    w0 = const["w0"]
    tau_1 = 0
    tau_2 = 0
    tau_3 = 0
    """
    dphi = x[3]
    dpsi = x[4]
    dth = x[5]
    """
    #w_0 = w0 * np.array([np.sin(th)*np.sin(psi)*np.cos(phi) - np.cos(th)*np.sin(phi),
    #                              np.cos(th)*np.sin(psi)*np.cos(phi) + np.sin(th)*np.sin(phi),
    #                              np.cos(psi)*np.cos(phi)])
    """
    w_1 = dphi*np.cos(psi)*np.sin(th) + dpsi*np.cos(th) + w_0[0]
    w_2 = dphi*np.cos(psi)*np.cos(th) - dpsi*np.sin(th) + w_0[1]
    w_3 = dth - dphi*np.sin(psi) + w_0[2]
    """
    if axis == 0:
        return (w1*np.sin(th) + w2*np.cos(th) - w0*np.sin(psi)*np.cos(phi))/np.cos(psi)
    elif axis == 1:
        return (w1*np.cos(th) - w2*np.sin(th) + w0*np.sin(phi))
    elif axis == 2:
        return (w1*np.sin(th) + w2*np.cos(th))*np.tan(psi) - w0*np.cos(phi)/np.cos(psi) + w3
    elif axis == 3:
        return (tau_1 - (I3-I2)*w3*w2)/I1
    elif axis == 4:
        return (tau_2 - (I1-I3)*w1*w3)/I2
    elif axis == 5:
        return (tau_3 - (I2-I1)*w2*w1)/I3

############# RK4 Function ##############

def RK4(t, x, const):
    # Necessary const parameters
    n = t.size
    t_step = t[1] - t[0]
    num_x = int(x.shape[0]) # Num of state variables

    K = np.zeros((num_x, 4))
    for i in range(1, n):
        for k in range(num_x):
            K[k, 0] = t_step * xDot(t[i-1], x[:, i-1], const, k)
            
            K[k, 1] = t_step * xDot(t[i-1] + t_step/2, x[:, i-1] + K[:, 0]/2, const, k)
            
            K[k, 2] = t_step * xDot(t[i-1] + t_step/2, x[:, i-1] + K[:, 1]/2, const, k)
            
            K[k, 3] = t_step * xDot(t[i-1] + t_step, x[:, i-1] + K[:, 2], const, k)
                
            x[k, i] = x[k, i - 1] + (K[k, 0] + 2 * (K[k, 1] + K[k, 2]) + K[k, 3]) / 6
        #for k in range(num_x):
        #    x[k+3, i] = xDot(t[i], np.concatenate((x[:3, i], x[3:, i-1]), axis=None), const, k)
        
        K.fill(0)
    return x

def scipy(t, x):
    phi = x[0]
    if np.cos(phi) == 0:
        phi += 1e-10
    psi = x[1]
    if np.cos(psi) == 0:
        psi += 1e-10
    th = x[2]
    w1 = x[3]
    w2 = x[4]
    w3 = x[5]
    I1 = 4500#const["I_sat"][0, 0]
    I2 = 6000#const["I_sat"][1, 1]
    I3 = 7000#const["I_sat"][2, 2]
    w0 = np.sqrt(3.986004354360959e5/(7000**3))
    tau_1 = 0
    tau_2 = 0
    tau_3 = 0
    return [(w1*np.sin(th) + w2*np.cos(th) - w0*np.sin(psi)*np.cos(phi))/np.cos(psi),
            (w1*np.cos(th) - w2*np.sin(th) + w0*np.sin(phi)),
            (w1*np.sin(th) + w2*np.cos(th))*np.tan(psi) - w0*np.cos(phi)/np.cos(psi) + w3,
            (tau_1 - (I3-I2)*w3*w2)/I1,
            (tau_2 - (I1-I3)*w1*w3)/I2,
            (tau_3 - (I2-I1)*w2*w1)/I3]
            