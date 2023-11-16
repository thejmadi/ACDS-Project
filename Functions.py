# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:09:23 2023

@author: tarun
"""

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

def SCIToECI():
    angle = np.radians(-23.5)
    return np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])

def ECIToECEF(angle):
    return np.array([[np.cos(angle), np.sin(angle), 0],
                    [-np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])

def ECIToBody(lat, i, Om, x):
    phi = x[0]
    psi = x[1]
    th = x[2]
    R_O_ECI_1 = np.array([[np.cos(lat), np.sin(lat), 0],
                          [-np.sin(lat), np.cos(lat), 0],
                          [0, 0, 1]])
    R_O_ECI_2 = np.array([[1, 0, 0],
                          [0, np.cos(i), np.sin(i)],
                          [0, -np.sin(i), np.cos(i)]])
    R_O_ECI_3 = np.array([[np.cos(Om), np.sin(Om), 0],
                          [-np.sin(Om), np.cos(Om), 0],
                          [0, 0, 1]])
    R_O_ECI = R_O_ECI_1 @ (R_O_ECI_2 @ R_O_ECI_3)
    
    R_B_O_1 = np.array([[np.cos(th), np.sin(th), 0],
                        [-np.sin(th), np.cos(th), 0],
                        [0, 0, 1]])
    R_B_O_2 = np.array([[1, 0, 0],
                        [0, np.cos(psi), np.sin(psi)],
                        [0, -np.sin(psi), np.cos(psi)]])
    R_B_O_3 = np.array([[np.cos(phi), np.sin(phi), 0],
                        [-np.sin(phi), np.cos(phi), 0],
                        [0, 0, 1]])
    
    R_B_O = R_B_O_1 @ (R_B_O_2 @ R_B_O_3)
    return R_B_O @ R_O_ECI
    
def GravGrad(w0, x, I):
    phi = x[0]
    psi = x[1]
    th = x[2]
    I1 = I[0, 0]
    I2 = I[1, 1]
    I3 = I[2, 2]
    GG = np.zeros(3)
    GG[0] = 3*(w0**2)*np.cos(psi)*np.sin(phi)*(np.cos(th)*np.sin(psi)*np.sin(phi)-np.sin(th)*np.cos(phi))*(I3-I2)
    GG[1] = 3*(w0**2)*np.cos(psi)*np.sin(phi)*(np.sin(th)*np.sin(psi)*np.sin(phi)+np.cos(th)*np.cos(phi))*(I1-I3)
    GG[2] = 3*(w0**2)*(np.cos(th)*np.cos(phi)+np.sin(th)*np.sin(psi)*np.sin(phi))*(np.cos(th)*np.sin(psi)*np.sin(phi)-np.sin(th)*np.cos(phi))*(I2-I1)
    return GG.reshape((3, 1))

def ElecMag(t, x, lat, const):
    # EM Torques
    w_e = t*7.29212e-5
    Re = 6371.2
    alp_m = np.radians(108.2)
    th_m = np.radians(169.7)
    g = (1e-9)*np.array([-29554.63, -1669.05, 5077.99]).reshape((3, 1))
    m_v = np.zeros((3, 1))
    m_v = (Re**3)*g
    m_hat = np.zeros((3, 1))
    r_hat = np.zeros((3, 1))
    B = np.zeros((3, 1))
    B_0 = 3.11e-5
    
    m_hat = (ECIToECEF(w_e).T @ (m_v/la.norm(m_v))).reshape((3, 1))
    r_hat = (x[6:9]/la.norm(x[6:9])).reshape((3, 1))
    B = B_0 * ((Re/la.norm(x[6:9]))**3)*(3*(m_hat[:, 0] @ r_hat)*r_hat - m_hat)
    
    B = ECIToBody(lat, const["i"], const["Om"], x[:3]) @ B
    EM = np.cross(const["M"], B.reshape(3))
    return EM

def SolPres(t, x, lat, const):
    SP = np.zeros(3)
    # Solar Torques
    w_year = 2*np.pi/365
    # Is SCIToECI correct?
    # Where is rho_a used
    s = ECIToBody(lat, const["i"], const["Om"], x) @ SCIToECI() @ np.array([[np.cos(w_year*t)], [np.sin(w_year*t)], [0]])
    s_hat = (s / la.norm(s)).reshape(3)
    
    for k in range(3):
        P = const["P_coeff"]
        A = const["plate_areas"][k]
        if(k != 0):
            r = const["plate_CoM"][k, :] - const["plate_CoM"][0, :]
        else:
            r = const["plate_CoM"][0, :]
        n = const["plate_norms"][k, :]
        n_hat = n / la.norm(n)
        rho_s = const["plate_spec"][k]
        rho_d = const["plate_diff"][k]
        #rho_a = const["plate_abs"][k]
        
        F_SP = P*A*np.dot(n_hat, s_hat)*((1-rho_s)*s_hat + (rho_s + 2*rho_d/3) * n_hat)
        SP += np.cross(r, F_SP)
    return SP

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
    lat = const["arg_per"] + w0*t
    
    GG = GravGrad(w0, x, const["I_sat"]).reshape(3)
    EM = ElecMag(t, x, lat, const)
    SP = SolPres(t, x, lat, const)
    tau = GG + EM + SP
    #tau = np.zeros(3)
    
    if axis == 0:
        return (w1*np.sin(th) + w2*np.cos(th) - w0*np.sin(psi)*np.cos(phi))/np.cos(psi)
    elif axis == 1:
        return (w1*np.cos(th) - w2*np.sin(th) + w0*np.sin(phi))
    elif axis == 2:
        return (w1*np.sin(th) + w2*np.cos(th))*np.tan(psi) - w0*np.cos(phi)/np.cos(psi) + w3
    elif axis == 3:
        return (tau[0] - (I3-I2)*w3*w2)/I1
    elif axis == 4:
        return (tau[1] - (I1-I3)*w1*w3)/I2
    elif axis == 5:
        return (tau[2] - (I2-I1)*w2*w1)/I3
    elif axis == 6:
        return x[9]
    elif axis == 7:
        return x[10]
    elif axis == 8:
        return x[11]
    elif axis == 9:
        return -const["mu"] * x[6] / la.norm(x[6:9])
    elif axis == 10:
        return -const["mu"] * x[7] / la.norm(x[6:9])
    elif axis == 11:
        return -const["mu"] * x[8] / la.norm(x[6:9])
    '''
    elif axis == 12:
        return GG[0]
    elif axis == 13:
        return GG[1]
    elif axis == 14:
        return GG[2]
    elif axis == 15:
        return EM[0]
    elif axis == 16:
        return EM[1]
    elif axis == 17:
        return EM[2]
    elif axis == 18:
        return SP[0]
    elif axis == 19:
        return SP[1]
    elif axis == 20:
        return SP[2]
    '''

############# RK4 Function ##############

def RK4(t, x, const):
    # Necessary const parameters
    n = t.size
    t_step = t[1] - t[0]
    num_x = int(x.shape[0]) # Num of state variables

    K = np.zeros((num_x, 4))
    for i in range(1, n):
        print(t[i])
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

############# Scipy Checker #############

def scipy(t, x):
    const = dict(a = 7000, i = np.radians(83), Om = np.radians(-70), f = np.radians(0), arg_per = np.radians(0),
                 D = 3, I_sat = np.diag([4500, 6000, 7000]), I_w = 0.5, plate_areas = np.array([5, 7, 7]),
                 plate_CoM = np.array([[0.1, 0.1, 0], [2, 0, 0], [-2, 0, 0]]), plate_abs = np.array([0.1, 0.7, 0.7]),
                 plate_spec = np.array([0.8, 0.2, 0.2]), plate_diff = np.array([0.1, 0.1, 0.1]),
                 mu = 3.986004354360959e5, P_coeff = 4.644e-6, beta = np.radians(30), M = np.array([3, 3, 3]))
    const["plate_norms"] = np.array([[0, 0, 1],
                                     [0, np.sin(const["beta"]), np.cos(const["beta"])],
                                     [0, np.sin(const["beta"]), np.cos(const["beta"])]])
    const["w0"] = np.sqrt(const["mu"]/const["a"]**3)
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
    lat = const["arg_per"] + w0*t
    
    #GG = GravGrad(w0, x, const["I"])
    EM = ElecMag(t, x, lat, const)
    #SP = SolPres(t, x, lat, const)
    tau = EM#GG + EM + SP
    return [(w1*np.sin(th) + w2*np.cos(th) - w0*np.sin(psi)*np.cos(phi))/np.cos(psi),
            (w1*np.cos(th) - w2*np.sin(th) + w0*np.sin(phi)),
            (w1*np.sin(th) + w2*np.cos(th))*np.tan(psi) - w0*np.cos(phi)/np.cos(psi) + w3,
            (tau[0] - (I3-I2)*w3*w2)/I1,
            (tau[1] - (I1-I3)*w1*w3)/I2,
            (tau[2] - (I2-I1)*w2*w1)/I3,
            x[9],
            x[10],
            x[11],
            -const["mu"] * x[6] / la.norm(x[6:9]),
            -const["mu"] * x[7] / la.norm(x[6:9]),
            -const["mu"] * x[8] / la.norm(x[6:9])]

############## Plotter ###############

def PlotAngles(t, x, title, xlab, ylab, labels):
    plt.rcParams['figure.dpi'] = 300
    c = ['b', 'g', 'r']
    for k in range(3):
        plt.plot(t, x[k], c = c[k], label = labels[k])
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    return