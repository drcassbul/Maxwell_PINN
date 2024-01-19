#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:16:29 2022

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
plt.rcParams.update({'font.size':15})

PI = np.pi

def PEC_cavity(x, y, t):
    w = PI*sqrt(2)
    Ez = sin(PI*x) * sin(PI*y) * cos(w*t)
    Hx = -1/sqrt(2) * sin(PI*x) * cos(PI*y) * sin(w*t)
    Hy = 1/sqrt(2) * cos(PI*x) * sin(PI*y) * sin(w*t)
    
    return Ez, Hx, Hy

# Grid dimensions
LX = 1
LY = 1

# Number of grid points
NX = 11
NY = 11

# Initialize spatial grid in physical units
GRID_X = np.linspace(0, LX, NX)
GRID_Y = np.linspace(0, LY, NY)

MESH_X, MESH_Y = np.meshgrid(GRID_X, GRID_Y, indexing='ij')

T = sqrt(2)
t = T
E_Z, H_X, H_Y = PEC_cavity(MESH_X, MESH_Y, t)


fig = plt.figure(figsize=(12, 4.3))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.3,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )

# Add data to image grid
grid[0].pcolormesh(MESH_X, MESH_Y, E_Z, vmin=-1, vmax=1)
grid[0].set_title(r'$E_z$')
grid[0].set_ylabel(r'$y$')
grid[1].pcolormesh(MESH_X, MESH_Y, H_X, vmin=-1, vmax=1)
grid[1].set_title(r'$H_x$')
im = grid[2].pcolormesh(MESH_X, MESH_Y, H_Y, vmin=-1, vmax=1)
grid[2].set_title(r'$H_y$')

for ax in grid:
    ax.axis('scaled')
    ax.set_xlabel(r'$x$')

# Colorbar
ax.cax.colorbar(im)
ax.cax.toggle_label(True)
plt.suptitle('$t=T$')
plt.savefig(f'PEC_cavity_figures/PEC_2D_t={t:.2f}.png')

#%%

# =============================================================================
# Temporal evolution in one point
# =============================================================================

x0 = 0.5
y0 = 0.5

t_grid = np.linspace(0, T, 200)
E_Z, H_X, H_Y = PEC_cavity(x0, y0, t_grid)

fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))


ax[0].plot(t_grid, E_Z, 'k-')
ax[0].plot(t_grid, H_X, 'b--')
ax[0].plot(t_grid, H_Y, 'r--')
ax[0].set_title(r'$(0.5, 0.5)$')
ax[0].set_ylabel(r'Field value')
ax[0].set_ylim(-1, 1)

E_Z, H_X, H_Y = PEC_cavity(0, y0, t_grid)

ax[1].plot(t_grid, E_Z, 'k-')
ax[1].plot(t_grid, H_X, 'b--')
ax[1].plot(t_grid, H_Y, 'r--')
ax[1].set_title(r'$(0, 0.5)$')
ax[1].set_ylim(-1, 1)


E_Z, H_X, H_Y = PEC_cavity(0.5, 0, t_grid)

ax[2].plot(t_grid, E_Z, 'k-', label='$E_z$')
ax[2].plot(t_grid, H_X, 'b--', label='$H_x$')
ax[2].plot(t_grid, H_Y, 'r--',label='$H_y$')
ax[2].set_title(r'$(0.5, 0)$')
ax[2].set_ylim(-1, 1)
plt.legend()

for ax in ax:
    ax.set_xlabel(r'$t$')

plt.tight_layout()
plt.savefig(f'PEC_cavity_figures/PEC_2D_time_dep.png')
