#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:03:57 2022

@author: rgelly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import axes3d
import scipy.constants as c
from numpy import cos, sin, sqrt, exp

PI = np.pi
# To use LaTeX code in the axis labels and title
plt.rc('font', **{'family' : 'serif', 'serif' : ['Computer Modern Roman']})
plt.rc('text', usetex = True)
#########
plt.rcParams.update({'font.size':18})


def plane_wave(x, t, k1, w):
    Ez = cos(k1*x - w*t)
    Hy = -cos(k1*x - w*t)
    
    return Ez, Hy

def plane_source_exp(t, t0, tau, w):
    Ez = exp(-(t-t0)**2/tau**2) * cos(w*t)
    Hy = -exp(-(t-t0)**2/tau**2) * cos(w*t)

    return Ez, Hy

def plane_source_door(t, tstart, tstop, w):
    if tstart < t < tstop:
        Ez = cos(w*t)
        Hy = -cos(w*t)
        return Ez, Hy
    else:
        return 0,0
    
def rect_media(x, y, x0, y0, dim_x, dim_y):
    MEDIA = np.where((np.abs(x - x0) <= dim_x) & (np.abs(y - y0) <= dim_y), 1, 0)
    return MEDIA

def periodic_slit(N_slits):
    slit_grid = np.linspace(0, 1, N_slits)
    pass

def moving_dielectric(t, x, y, r_cyl, w_cyl):
    x0 = sin(w_cyl*t) + 0.5
    MEDIA = np.where(((x - x0)**2 + (y - 0.5)**2 <= r_cyl**2), 1, 0) # Cylinder
    return MEDIA

# Dimensions of grid
LX = 1
LY = 1

# Number of grid points
NX = 1001
NY = 1001

# Space increments
DX = LX / (NX - 1)
DY = LY / (NY - 1)

# Time increment (CFL condition)
DT = 1/sqrt(1/DX**2 + 1/DY**2) /5

w = 2*np.pi * 4
eps1 = 1
mu1 = 1
k1 = w*sqrt(eps1*mu1)


# Total integration time
Tp = 2*PI/w # Time period
T = 15 * Tp
NT = int(T/DT)

# Initialize spatial grid in physical units
GRID_X = np.linspace(0, LX, NX)
GRID_Y = np.linspace(0, LY, NY)

# Staggered grid for Ez (NX-1 x NY-1)
MESH_X, MESH_Y = np.meshgrid(0.5*(GRID_X[1:] + GRID_X[:-1]), 0.5*(GRID_Y[1:] + GRID_Y[:-1]), indexing='ij')

# Grid for Hx (NX - 1  x NY)
MESH_HX_X, MESH_HX_Y = np.meshgrid(0.5*(GRID_X[1:] + GRID_X[:-1]), GRID_Y, indexing='ij')

# Grid for Hy (NX - 1 x NY)
MESH_HY_X, MESH_HY_Y = np.meshgrid(GRID_X, 0.5*(GRID_Y[1:] + GRID_Y[:-1]), indexing='ij')

FULL_MESH_X, FULL_MESH_Y = np.meshgrid(np.linspace(0, (NX-1)*DX, 2*NX-1), np.linspace(0, (NY-1)*DY, 2*NY-1), indexing='ij')


PERMITTIVITY = np.array([1, 1])
PERMEABILITY = np.array([1, 1])
ELECTRIC_CONDUCTIVITY = np.array([0, 1e8])
MAGNETIC_LOSS = np.array([0, 0])


# Compute coefficients for E-Field components
C_A = ((1 - 0.5 * ELECTRIC_CONDUCTIVITY * DT / PERMITTIVITY) /
       (1 + 0.5 * ELECTRIC_CONDUCTIVITY * DT / PERMITTIVITY))

C_Bx = ((DT / (PERMITTIVITY * DX)) /
       (1 + 0.5 * ELECTRIC_CONDUCTIVITY * DT / PERMITTIVITY))

C_By = ((DT / (PERMITTIVITY * DY)) /
       (1 + 0.5 * ELECTRIC_CONDUCTIVITY * DT / PERMITTIVITY))

# Compute coefficients for H-Field components
D_A = ((1 - 0.5 * MAGNETIC_LOSS * DT / PERMEABILITY) /
       (1 + 0.5 * MAGNETIC_LOSS * DT / PERMEABILITY))

D_Bx = ((DT / (PERMEABILITY * DX)) /
       (1 + 0.5 * MAGNETIC_LOSS * DT / PERMEABILITY))

D_By = ((DT / (PERMEABILITY * DY)) /
       (1 + 0.5 * MAGNETIC_LOSS * DT / PERMEABILITY))

MEDIA = np.full_like(FULL_MESH_X, 0)
#MEDIA = np.where(((FULL_MESH_X - 0.5)**2 + (FULL_MESH_Y - 0.5)**2 <= 0.25**2), 1, 0) # Cylinder
# MEDIA = np.where(((FULL_MESH_X - 0.8)**2 + (FULL_MESH_Y - 0.8)**2 <= 0.15**2), 1, 0) # Cylinder
# MEDIA += np.where(((FULL_MESH_X - 0.8)**2 + (FULL_MESH_Y - 0.2)**2 <= 0.15**2), 1, 0) # Cylinder
# MEDIA += np.where((np.abs(FULL_MESH_X - 0.5) <= 0.3) & (np.abs(FULL_MESH_Y - 0.5) <= 0.15), 1, 0)


x0 = 0.5
y0=0.5
dim_x = 0.1
dim_y = 0.2
#MEDIA = np.where((np.abs(FULL_MESH_X - x0) <= dim_x) & (np.abs(FULL_MESH_Y - y0) <= dim_y), 1, 0)

plt.figure()
plt.pcolormesh(MEDIA)
plt.axis('scaled')

C_A_grid = np.full_like(MEDIA, np.nan, dtype = float)
C_Bx_grid = np.full_like(MEDIA, np.nan, dtype = float)
C_By_grid = np.full_like(MEDIA, np.nan, dtype = float)
D_A_grid = np.full_like(MEDIA, np.nan, dtype = float)
D_Bx_grid = np.full_like(MEDIA, np.nan, dtype = float)
D_By_grid = np.full_like(MEDIA, np.nan, dtype = float)

for m in range(2):
    idx_media = (MEDIA == m)
    C_A_grid[idx_media] = C_A[m]
    C_Bx_grid[idx_media] = C_Bx[m]
    C_By_grid[idx_media] = C_By[m]
    D_A_grid[idx_media] = D_A[m]
    D_Bx_grid[idx_media] = D_Bx[m]
    D_By_grid[idx_media] = D_By[m]
    
    
E_Z = np.zeros_like(MESH_X)
H_X = np.zeros_like(MESH_HX_X) 
H_Y = np.zeros_like(MESH_HY_X) 

# =============================================================================
# Initial conditions
# =============================================================================

# Plane wave propagating towards the x direction
# E_Z, _, = plane_wave(MESH_X, -0.5*DT, k1, w)
# _, H_Y = plane_wave(MESH_HY_X, 0, k1, w)
# H_X = np.zeros_like(MESH_HX_X)

# E_Z[MESH_X > 0.25] = 0
# H_Y[MESH_HY_X > 0.25] = 0
# E_Z[np.abs(MESH_Y-0.5) > 0.3] = 0
# H_Y[np.abs(MESH_HY_Y-0.5) > 0.3] = 0

# # Electrical field inside the obstacle = 0
# E_Z[(MESH_X - 0.5)**2 + (MESH_Y - 0.5)**2 <= 0.25**2] = 0


# =============================================================================
# Yee's algorithm
# =============================================================================

i_snap = 0
plt.ioff()

E_Z_inc = np.zeros_like(MESH_X)
H_Y_inc = np.zeros_like(MESH_HY_X) 

for n in range(NT):
        
    E_Z = C_A_grid[1::2, 1::2] * E_Z + C_Bx_grid[1::2, 1::2] * (H_Y[1:, :] - H_Y[:-1, :]) + C_By_grid[1::2, 1::2] * (H_X[:, :-1] - H_X[:, 1:])
    H_X[:, 1:-1] = D_A_grid[1::2, 0::2][:, 1:-1] * H_X[:, 1:-1] + D_By_grid[1::2, 0::2][:, 1:-1] * (E_Z[:, :-1] - E_Z[:, 1:])
    H_Y[1:-1, :] = D_A_grid[0::2, 1::2][1:-1, :] * H_Y[1:-1, :] + D_Bx_grid[0::2, 1::2][1:-1, :] * (E_Z[1:, :] - E_Z[:-1, :])

    # Boundary conditions for the magnetic field
    # 1/2 time step
    E_Z_half = C_A_grid[1::2, 1::2] * E_Z + 0.5*C_Bx_grid[1::2, 1::2] * (H_Y[1:, :] - H_Y[:-1, :]) + 0.5*C_By_grid[1::2, 1::2] * (H_X[:, :-1] - H_X[:, 1:])
    H_X[:, 0] = E_Z_half[:, 0] 
    H_X[:, -1] = -E_Z_half[:, -1]
    H_Y[0, :] = E_Z_half[0, :] - plane_source_door(t=n*DT, tstart=0, tstop=2*Tp, w=w)[0] + plane_source_door(t=n*DT, tstart=0, tstop=2*Tp, w=w)[1]
    H_Y[-1, :] = -E_Z_half[-1, :]

    # Boundary conditions for the electric field (Silver-Muller)
    # E_Z[:, 0] = -H_X[:, 0]
    # E_Z[:, -1] = H_X[:, -1]
    
    # E_Z[0, :] = H_Y[0, :] 
    # E_Z[0, :] = H_Y[0, :] - plane_source_exp(t=n*DT, t0=4*Tp, tau=4*Tp, w=w*2)[1] + plane_source_exp(t=n*DT, t0=4*Tp, tau=4*Tp, w=w*2)[0]
    # E_Z[-1, :] = -H_Y[-1, :]
    
    # Electric field source
    # E_Z[0,:] = plane_source_door(t=n*DT, tstart=0, tstop=T, w=w)[0]
    # H_Y[0, :] = plane_source_door(t=n*DT, tstart=0, tstop=T, w=w)[1]
    # H_X[0,:] = 0
    #E_Z[0, :] = plane_source_door(t=n*DT, tstart=0, tstop=2*Tp, w=w)[0]
    #H_Y[NX//2, :] = plane_source_door(t=n*DT, tstart=0, tstop=2*Tp, w=w)[1]
    
    if n % 50 == 0:
        plt.figure()
        plt.pcolormesh(MESH_X, MESH_Y, E_Z, vmin=-1, vmax=1, cmap='seismic')
        plt.axis('scaled')
        plt.colorbar()
        plt.title(f"$t={n*DT:.2f}$")
        plt.tight_layout()
        plt.savefig(f"ABC_diffraction/snapshots/num_snap_{i_snap}.png")       
        plt.close()
        i_snap+=1
        
        
