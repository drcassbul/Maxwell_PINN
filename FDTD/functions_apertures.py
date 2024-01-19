#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:08:51 2022

@author: rgelly
"""
import numpy as np

def rectangle_material(x, y, pos, dim, m_in, m_out):
    return np.where((np.abs(x - pos[0]) < dim[0]/2) & (np.abs(y - pos[1]) < dim[0]/2), m_in, m_out)

def pupil(x, y, d, m_in, m_out): #Inidividual pupil
    return np.where(x**2+y**2 <= d**2/4, m_in, m_out)

def Ncircular(x, y, d, N_pupils, r_circle, pos_center, m_in, m_out):   
    P = np.zeros((len(x), len(y)))
    for n in range(N_pupils):
        Z = r_circle*np.exp(1j*2*np.pi*n/N_pupils)
        xn = np.real(Z) + pos_center[0]
        yn = np.imag(Z) + pos_center[1]
        P += pupil(x-xn, y-yn, d, m_in, m_out)
    return P