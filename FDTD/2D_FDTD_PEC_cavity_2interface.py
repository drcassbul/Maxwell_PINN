#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:00:59 2022

@author: rgelly
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import axes3d
import scipy.constants as c
from numpy import cos, sin, sqrt


# To use LaTeX code in the axis labels and title
plt.rc('font', **{'family' : 'serif', 'serif' : ['Computer Modern Roman']})
plt.rc('text', usetex = True)
#########
plt.rcParams.update({'font.size':18})

PI = np.pi

def PEC_2_interfaces(x, y, t, w, wy, eps1, eps2):
    Ez = np.full_like(x, np.nan)
    Hx = np.full_like(x, np.nan)
    Hy = np.full_like(x, np.nan)
    
    i_inter1 = np.argmin(np.abs(x[:,0] + 0.5))
    i_inter2 = np.argmin(np.abs(x[:,0] - 0.5))
    
    w1 = sqrt(eps1*w**2 - wy**2)
    w2 = sqrt(eps2*w**2 - wy**2)
    
    for i in range(i_inter1):
        Ez[i] = sin(0.5*w2) * sin(w1*(x[i]+1)) * sin(wy*y[i]) * cos(w*t)
        Hx[i] = -wy/w * sin(0.5*w2) * sin(w1*(x[i]+1)) * cos(wy*y[i]) * sin(w*t)
        Hy[i] = w1/w * sin(0.5*w2) * cos(w1*(x[i]+1)) * sin(wy*y[i]) * sin(w*t)
    
    for i in range(i_inter1, i_inter2):
        Ez[i] = -sin(0.5*w1) * sin(w2*x[i]) * sin(wy*y[i]) * cos(w*t)
        Hx[i] = wy/w * sin(0.5*w1) * sin(w2*x[i]) * cos(wy*y[i]) * sin(w*t)
        Hy[i] = -w2/w * sin(0.5*w1) * cos(w2*x[i]) * sin(wy*y[i]) * sin(w*t)
        
    for i in range(i_inter2, x.shape[0]):
        Ez[i] = sin(0.5*w2) * sin(w1*(x[i]-1)) * sin(wy*y[i]) * cos(w*t)
        Hx[i] = -wy/w * sin(0.5*w2) * sin(w1*(x[i]-1)) * cos(wy*y[i]) * sin(w*t)
        Hy[i] = w1/w * sin(0.5*w2) * cos(w1*(x[i]-1)) * sin(wy*y[i]) * sin(w*t)
    
    return Ez, Hx, Hy
        
NX = 201
NY = 201

Tp = 2/sqrt(5)

# Initialize spatial grid in physical units
GRID_X = np.linspace(-1, 1, NX)
GRID_Y = np.linspace(-1, 1, NY)

# Staggered grid for Ez (NX-1 x NY-1)
MESH_X, MESH_Y = np.meshgrid(GRID_X, GRID_Y, indexing='ij')

t  = 1
Ez, Hx, Hy = PEC_2_interfaces(MESH_X, MESH_Y, t=t, w=9.07716175885174, wy=2*PI, eps1=1, eps2=2.25)

plt.figure()
plt.pcolormesh(MESH_X, MESH_Y, Ez, cmap='seismic')
plt.axis('scaled')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$E_z$')
plt.colorbar()
plt.tight_layout()
plt.savefig('PEC_cavity_interface_figures/PEC_interface_Ez.png')


fig, ax = plt.subplots(1, 3, figsize=(15, 5))



# Add data to image grid
ax0 = ax[0].pcolormesh(MESH_X, MESH_Y, Ez, cmap='seismic')
ax[0].set_title(r'$E_z$')
ax[0].set_ylabel(r'$y$')
fig.colorbar(ax0, ax=ax[0])
ax1 = ax[1].pcolormesh(MESH_X, MESH_Y, Hx, cmap='seismic')
ax[1].set_title(r'$H_x$')
fig.colorbar(ax1, ax=ax[1])
ax2 = im = ax[2].pcolormesh(MESH_X, MESH_Y, Hy, cmap='seismic')
ax[2].set_title(r'$H_y$')
fig.colorbar(ax2, ax=ax[2])

for ax in ax:

    ax.axis('scaled')
    ax.set_xlabel(r'$x$')
        