# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:16:35 2023

@author: tarun
"""
import Functions as fc
import numpy as np
from scipy import integrate as sci


def main():
    const = dict(a = 7000, i = np.radians(83), Om = np.radians(-70), f = np.radians(0), arg_per = np.radians(0),
                 D = 3, I_sat = np.diag([4500, 6000, 7000]), I_w = 0.5, plate_areas = np.array([5, 7, 7]),
                 plate_CoM = np.array([[0.1, 0.1, 0], [2, 0, 0], [-2, 0, 0]]), plate_abs = np.array([0.1, 0.7, 0.7]),
                 plate_spec = np.array([0.8, 0.2, 0.2]), plate_diff = np.array([0.1, 0.1, 0.1]),
                 mu = 3.986004354360959e5, P_coeff = 4.644e-6, beta = np.radians(30), M = np.array([[3, 3, 3]]))
    const["plate_norms"] = np.array([[0, 0, 1],
                                     [0, np.sin(const["beta"]), np.cos(const["beta"])],
                                     [0, np.sin(const["beta"]), np.cos(const["beta"])]])
    const["w0"] = np.sqrt(const["mu"]/const["a"]**3)
    Period = 2*np.pi*np.sqrt((const['a']**3)/const["mu"])
    t = np.arange(0, 3*Period, 0.1)
    a = const["a"]
    mu = const["mu"]
    i = const["i"]
    
    x = np.zeros((6, t.size))
    #torques = np.zeros((9, t.size))
    torques = []
    torque_t = []
    x[:, 0] = np.array([-0.07, 0.09,  0.15,                                     # Angles
                        -8e-4, 6e-4, 8.5e-4])                                    # Angle Rates
                        #a, 0, 0,                                                # Pos
                        #0, np.cos(i)*np.sqrt(mu/a), np.sin(i)*np.sqrt(mu/a),])    # Vel
    '''
                        0, 0, 0,                                                # GG
                        0, 0, 0,                                                # EM
                        0, 0, 0])                                               # SP
    '''
    #x, torques = fc.RK4(t, x, const, torques)
    res = sci.solve_ivp(fc.scipy, [0, 3*Period], x[:, 0], t_eval=t, args=[torques, torque_t], max_step=10)
    print()
    fc.PlotState(res.t/Period, res.y)
    fc.PlotTorques(torque_t/Period, torques)
    #fc.PlotTorques(t/Period, torques)
    #fc.PlotState(t/Period, x)
    #fc.PlotTorques(t/Period, torques)
    #fc.PlotAngles(t/Period, x[12:15, :], title[2], xlab[0], ylab[2], labels[3:])
    #fc.PlotAngles(t/Period, x[15:18, :], title[3], xlab[0], ylab[2], labels[3:])
    #fc.PlotAngles(t/Period, x[18:21, :], title[4], xlab[0], ylab[2], labels[3:])

main()