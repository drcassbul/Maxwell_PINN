#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:49:05 2022

@author: rgelly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.constants as c
from numpy import cos, sin, sqrt

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
MAGNETIC_LOSS_VACCUM = 0 # Equivalent magnetic loss in vacuum

# =============================================================================
# Numerical parameters
# =============================================================================

# Mesh parameters
DX = 0.1
DY = DX 

# Number of grid points
NX = 500
NY = 500

# Physical dimensions of grid
LX = (NX-1) * DX
LY = (NY-1) * DY

# Time increment
DT = 1e-10

# Number of time steps
NT = 500

# Initialize spatial grid in physical units
GRID_X = np.arange(0, NX*DX, DX)
GRID_Y = np.arange(0, NY*DX, DY)

# Staggered grid for Ez (NX-1 x NY-1)
MESH_X, MESH_Y = np.meshgrid(0.5*(GRID_X[1:] + GRID_X[:-1]), 0.5*(GRID_Y[1:] + GRID_Y[:-1]), indexing='ij')

# Grid for Hx (NX - 1  x NY)
MESH_HX_X, MESH_HX_Y = np.meshgrid(0.5*(GRID_X[1:] + GRID_X[:-1]), GRID_Y, indexing='ij')

# Grid for Hy (NX - 1 x NY)
MESH_HY_X, MESH_HY_Y = np.meshgrid(GRID_X, 0.5*(GRID_Y[1:] + GRID_Y[:-1]), indexing='ij')

FULL_MESH_X, FULL_MESH_Y = np.meshgrid(np.arange(0, (NX-0.5)*DX, 0.5*DX), np.arange(0, (NY-0.5)*DY, 0.5*DY))


# =============================================================================
# Set up algorithm
# =============================================================================


# Integer array storing media type at each field component location
N_MEDIA = 1
MEDIA = np.full((NX*2-1, NY*2-1), 0)


# Arrays of the physical properties of the material at each grid point
# m=0: vacuum

PERMITTIVITY = np.array([PERMITTIVITY_VACUUM])
PERMEABILITY = np.array([PERMEABILITY_VACUUM])
ELECTRIC_CONDUCTIVITY = np.array([ELECTRIC_CONDUCTIVITY_VACUUM])
MAGNETIC_LOSS = np.array([MAGNETIC_LOSS_VACCUM])


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

E_Z, _, _ = PEC_cavity(MESH_X, MESH_Y, -0.5*DT, LX, LY)
_, H_X, _ = PEC_cavity(MESH_HX_X, MESH_HX_Y, 0, LX, LY)
_, _, H_Y = PEC_cavity(MESH_HY_X, MESH_HY_Y, 0, LX, LY)


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


for n in range(NT):
    E_Z = C_A_vac * E_Z + C_B_vac * (H_Y[1:, :] - H_Y[:-1, :] + H_X[:, 1:] - H_X[:, :-1])
    
    H_X_new = np.zeros_like(H_X)
    H_Y_new = np.zeros_like(H_Y)
    
    H_X_new[:, 1:-1] = D_A_vac * H_X[:, 1:-1] + D_B_vac * (E_Z[:, 1:] - E_Z[:, :-1])
    H_Y_new[1:-1, :] = D_A_vac * H_Y[1:-1, :] + D_B_vac * (E_Z[1:, :] - E_Z[:-1, :])

    # Perfect Magnetic Conductor (PMC) conditions
    H_X_new[0, :] = 0
    H_X_new[-1, :] = 0
    H_Y_new[:, 0] = 0
    H_Y_new[:, -1] = 0
    
    
    E_Z[:, 0] = 0
    E_Z[:, -1] = 0
    E_Z[0, :] = 0
    E_Z[-1, :] = 0
    
    H_X = H_X_new
    H_Y = H_Y_new

    # Analytical solution
    E_Z_ref, _, _ = PEC_cavity(MESH_X, MESH_Y, (n+0.5)*DT, LX, LY)
    _, H_X_ref, _ = PEC_cavity(MESH_HX_X, MESH_HX_Y, (n+1)*DT, LX, LY)
    _, _, H_Y_ref = PEC_cavity(MESH_HY_X, MESH_HY_Y, (n+1)*DT, LX, LY)
    
    E_Z_err = (E_Z - E_Z_ref) / E_Z_ref
    H_X_err = (H_X - H_X_ref) / H_X_ref
    H_Y_err = (H_Y - H_Y_ref) / H_Y_ref
    
    if n % 10 == 0:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        ax.plot_wireframe(MESH_HX_X, MESH_HX_Y, H_X_ref, rstride=2, cstride=2)
        ax.set_zlim(-1, 1)
        plt.savefig(f"2D_FDTD_snapshots/exact_snap_{n}.png")       
        plt.close()
        
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        ax.plot_wireframe(MESH_X, MESH_Y, E_Z, rstride=2, cstride=2)
        ax.set_zlim(-1, 1)
        plt.savefig(f"2D_FDTD_snapshots/num_snap_{n}.png")       
        plt.close()
        
        plt.figure()
        plt.pcolormesh(MESH_X, MESH_Y, E_Z, vmin=-1, vmax=1)
        plt.axis('scaled')
        plt.colorbar()
        plt.savefig(f"2D_FDTD_snapshots/snapshots_color/snap_{n}.png")       
        plt.close()
        