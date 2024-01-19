#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:20:00 2022

@author: rgelly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.constants as c
from numpy import cos, sin, sqrt
from functions_apertures import rectangle_material, Ncircular

PI = np.pi

def PEC_cavity(x, y, t, LX, LY):
    w = c.c * sqrt((PI/LX)**2 + (PI/LY)**2)
    Ez = sin(PI*x/LX) * sin(PI*y/LY) * cos(w*t)
    Hx = -1/sqrt(2) * sin(PI*x/LX) * cos(PI*y/LY) * sin(w*t)
    Hy = 1/sqrt(2) * cos(PI*x/LX) * sin(PI*y/LY) * sin(w*t)
    
    return Ez, Hx, Hy

# =============================================================================
# Physical constants
# =============================================================================

PERMITTIVITY_VACUUM = c.epsilon_0 # Electric permittivity of vacuum
PERMEABILITY_VACUUM = c.mu_0 # Permeability of vacuum
ELECTRIC_CONDUCTIVITY_VACUUM = 0 # Electric conductivity in vacuum
ELECTRIC_CONDUCTIVITY_GLASS = 0
MAGNETIC_LOSS_VACCUM = 0 # Equivalent magnetic loss in vacuum
RELATIVE_PERMITTIVITY_GLASS = 5
RELATIVE_PERMEABILITY_GLASS = 3/8


# =============================================================================
# Numerical parameters
# =============================================================================

# Mesh parameters
DX = 0.1
DY = DX 

# Number of grid points
NX = 100
NY = 100

# Physical dimensions of grid
LX = (NX-1) * DX
LY = (NY-1) * DY


# Time increment
DT = 5e-11

# Number of time steps
NT = 500

# Initialize spatial grid in physical units
GRID_X = np.arange(0, NX*DX, DX)
GRID_Y = np.arange(0, NY*DX, DY)

# Staggered grid for Ez (NX-1 x NY-1)
MESH_X, MESH_Y = np.meshgrid(0.5*(GRID_X[1:] + GRID_X[:-1]), 0.5*(GRID_Y[1:] + GRID_Y[:-1]))
FULL_MESH_X, FULL_MESH_Y = np.meshgrid(np.arange(0, (NX-0.5)*DX, 0.5*DX), np.arange(0, (NY-0.5)*DY, 0.5*DY))


# =============================================================================
# Set up algorithm
# =============================================================================


# Integer array storing media type at each field component location
N_MEDIA = 2
MEDIA = np.full((NX*2-1, NY*2-1), 0)
# MEDIA = rectangle_material(FULL_MESH_X, FULL_MESH_Y, (NX*DX/2, NY*DY/2), (NX*DX/2, NY*DY/2), 1, 0)
# MEDIA -= rectangle_material(FULL_MESH_X, FULL_MESH_Y, (NX*DX/2, NY*DY/2), (NX*DX/2-1, NY*DY/2-1), 1, 0)
# MEDIA += rectangle_material(FULL_MESH_X, FULL_MESH_Y, (NX*DX/2, NY*DY/2), (NX*DX/2-2, NY*DY/2-2), 1, 0)
# MEDIA -= rectangle_material(FULL_MESH_X, FULL_MESH_Y, (NX*DX/2, NY*DY/2), (NX*DX/2-3, NY*DY/2-3), 1, 0)

# MEDIA = Ncircular(FULL_MESH_X, FULL_MESH_Y, NX*DX/10, 10, NX*DX/3, (NX*DX*0.5, NX*DX*0.5), 1, 0)

plt.figure()
plt.pcolormesh(MEDIA)
plt.axis('scaled')

# Arrays of the physical properties of the material at each grid point
# m=0: vacuum
# m=1: glass
PERMITTIVITY = np.array([PERMITTIVITY_VACUUM, PERMITTIVITY_VACUUM*RELATIVE_PERMITTIVITY_GLASS])
PERMEABILITY = np.array([PERMEABILITY_VACUUM, PERMEABILITY_VACUUM*RELATIVE_PERMEABILITY_GLASS])
ELECTRIC_CONDUCTIVITY = np.array([ELECTRIC_CONDUCTIVITY_VACUUM, ELECTRIC_CONDUCTIVITY_GLASS])
MAGNETIC_LOSS = np.array([MAGNETIC_LOSS_VACCUM, MAGNETIC_LOSS_VACCUM])


# Compute coefficients for E-Field components
C_A = ((1 - 0.5 * ELECTRIC_CONDUCTIVITY * DT / PERMITTIVITY) /
       (1 + 0.5 * ELECTRIC_CONDUCTIVITY * DT / PERMITTIVITY))

C_B = ((DT / (PERMITTIVITY * DX)) /
       (1 + 0.5 * ELECTRIC_CONDUCTIVITY * DT / PERMITTIVITY))

# Compute coefficients for H-Field components
D_A = ((1 - 0.5 * MAGNETIC_LOSS * DT / PERMEABILITY) /
       (1 + 0.5 * MAGNETIC_LOSS * DT / PERMEABILITY))

D_B = ((DT / (PERMEABILITY * DX)) /
       (1 + 0.5 * MAGNETIC_LOSS * DT / PERMEABILITY))

# =============================================================================
# Initial conditions
# =============================================================================

H_X = np.zeros((NX-1, NY))
H_Y = np.zeros((NX, NY-1))

kx = 2*np.pi / ((NX-1)*DX)
ky = 2*np.pi / ((NY-1)*DY)
#E_Z = np.cos(kx*MESH_X) * np.cos(ky*MESH_Y)
E_Z, _, _ = PEC_cavity(MESH_X, MESH_Y, -0.5*DT, LX, LY)


fig = plt.figure()
ax = axes3d.Axes3D(fig)
ax.plot_wireframe(MESH_X, MESH_Y, E_Z, rstride=2, cstride=2)

# =============================================================================
# Yee's algorithm
# =============================================================================

C_A_vac = C_A[0]
C_B_vac = C_B[0]
D_A_vac = D_A[0]
D_B_vac = D_B[0]

C_A_grid = np.full_like(MEDIA, np.nan, dtype = float)
C_B_grid = np.full_like(MEDIA, np.nan, dtype = float)
D_A_grid = np.full_like(MEDIA, np.nan, dtype = float)
D_B_grid = np.full_like(MEDIA, np.nan, dtype = float)

for m in range(N_MEDIA):
    idx_media = (MEDIA == m)
    C_A_grid[idx_media] = C_A[m]
    C_B_grid[idx_media] = C_B[m]
    D_A_grid[idx_media] = D_A[m]
    D_B_grid[idx_media] = D_B[m]


for n in range(NT):
    # E_Z = C_A_vac * E_Z + C_B_vac * (H_Y[1:, :] - H_Y[:-1, :] + H_X[:, 1:] - H_X[:, :-1])
    # H_X_new = np.full_like(H_X, np.nan)
    # H_Y_new = np.full_like(H_Y, np.nan)
    # H_X_new[:, 1:-1] = D_A_vac * H_X[:, 1:-1] + D_B_vac * (E_Z[:, 1:] - E_Z[:, :-1])
    # H_Y_new[1:-1, :] = D_A_vac * H_Y[1:-1, :] + D_B_vac * (E_Z[1:, :] - E_Z[:-1, :])
    
    # # Periodic boundary conditions
    # H_X_new[:, 0] = D_A_vac * H_X[:, 0] + D_B_vac * (E_Z[:, 0] - E_Z[:, -1])
    # H_X_new[:, -1] = H_X_new[:, 0]
    # H_Y_new[0, :] = D_A_vac * H_Y[0, :] + D_B_vac * (E_Z[0, :] - E_Z[-1, :])
    # H_Y_new[-1, :] = H_Y_new[0, :]
    
    # H_X = H_X_new
    # H_Y = H_Y_new

    E_Z = C_A_grid[1::2, 1::2] * E_Z + C_B_grid[1::2, 1::2] * (H_Y[1:, :] - H_Y[:-1, :] + H_X[:, 1:] - H_X[:, :-1])
    H_X_new = np.full_like(H_X, np.nan)
    H_Y_new = np.full_like(H_Y, np.nan)
    H_X_new[:, 1:-1] = D_A_grid[1::2, 0::2][:, 1:-1] * H_X[:, 1:-1] + D_B_grid[1::2, 0::2][:, 1:-1] * (E_Z[:, 1:] - E_Z[:, :-1])
    H_Y_new[1:-1, :] = D_A_grid[0::2, 1::2][1:-1, :] * H_Y[1:-1, :] + D_B_grid[0::2, 1::2][1:-1, :] * (E_Z[1:, :] - E_Z[:-1, :])
    
    # Periodic boundary conditions
    # H_X_new[:, 0] = D_A_vac * H_X[:, 0] + D_B_vac * (E_Z[:, 0] - E_Z[:, -1])
    # H_X_new[:, -1] = H_X_new[:, 0]
    # H_Y_new[0, :] = D_A_vac * H_Y[0, :] + D_B_vac * (E_Z[0, :] - E_Z[-1, :])
    # H_Y_new[-1, :] = H_Y_new[0, :]
    
    # Perfect Magnetic Conductor (PMC) conditions
    H_X_new[:, 0] = 0
    H_X_new[:, -1] = 0
    H_Y_new[0, :] = 0
    H_Y_new[-1, :] = 0
    
    H_X = H_X_new
    H_Y = H_Y_new
    if n % 10 == 0:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        ax.plot_wireframe(MESH_X, MESH_Y, E_Z, rstride=2, cstride=2)
        ax.set_zlim(-1, 1)
        plt.savefig(f"2D_FDTD_snapshots/snap_{n}.png")       
        plt.close()
        
        plt.figure()
        plt.pcolormesh(MESH_X, MESH_Y, E_Z, vmin=-1, vmax=1)
        plt.axis('scaled')
        plt.colorbar()
        plt.savefig(f"2D_FDTD_snapshots/snapshots_color/snap_{n}.png")       
        plt.close()
        
                
            
            