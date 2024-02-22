#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:49:05 2022

@author: rgelly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import StrMethodFormatter
import scipy.constants as c
from numpy import cos, sin, sqrt
import os

# To use LaTeX code in the axis labels and title
plt.rc('font', **{'family' : 'serif', 'serif' : ['Computer Modern Roman']})
plt.rc('text', usetex = True)
#########
plt.rcParams.update({'font.size':15})

PI = np.pi

def plane_wave(x, t, k, w):
    Ez = cos(k*x - w*t)
    Hy = -cos(k*x - w*t)
    
    return Ez, Hy

def plane_source_door(t, tstart, tstop, w):
    if tstart < t < tstop:
        Ez = cos(w*t)
        Hy = -cos(w*t)
        return Ez, Hy
    else:
        return 0,0

def plane_source_exp(t, t0, tau, w):
    Ez = np.exp(-(t-t0)**2/tau**2) * cos(w*t)
    Hy = -np.exp(-(t-t0)**2/tau**2) * cos(w*t)

    return Ez, Hy

def rect_scatterer(x, y, x0, y0, dim_x, dim_y, eps_2, eps_1):
    eps_grid = np.where((np.abs(x - x0) <= dim_x) & (np.abs(y - y0) <= dim_y), eps_2, eps_1)
    return eps_grid

# =============================================================================
# Physical constants
# =============================================================================

eps_1 = 1 # Relative permittivity of medium
mu_1 = 1 # Relative permeability of medium
eps_2 = 1
mu_2 = 1
c_r = 1/np.sqrt(eps_1*mu_1) # Relative speed of light in medium

# =============================================================================
# Numerical parameters
# =============================================================================

# Dimensions of grid
LX = 1
LY = 1

# Number of grid points
NX = 301
NY = 301

# Space increments
DX = LX / (NX - 1)
DY = LY / (NY - 1)

# Time increment (CFL condition)
DT = 1 / (c_r * sqrt(1/DX**2 + 1/DY**2))

# Total integration time
Tp = sqrt(2) # Time period
T = 10 * Tp
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


# =============================================================================
# Initial conditions
# =============================================================================

E_Z = np.zeros_like(MESH_X)
H_X = np.zeros_like(MESH_HX_X) 
H_Y = np.zeros_like(MESH_HY_X) 


# =============================================================================
# Yee's algorithm
# =============================================================================

E_Z_array = np.full((NT, *E_Z.shape), np.nan)
H_X_array = np.full((NT, *H_X.shape), np.nan)
H_Y_array = np.full((NT, *H_Y.shape), np.nan)

E_Z_ref_array = np.full((NT, *E_Z.shape), np.nan)
H_X_ref_array = np.full((NT, *H_X.shape), np.nan)
H_Y_ref_array = np.full((NT, *H_Y.shape), np.nan)

# Partition the grid into two media
eps_grid = np.where(MESH_X < 0.5, eps_1, eps_2)
mu_grid_HX = np.where(MESH_HX_X < 0.5, mu_1, mu_2)
mu_grid_HY = np.where(MESH_HY_Y < 0.5, mu_1, mu_2)


for n in range(NT):
    E_Z = E_Z + (DT / eps_grid) * ((H_Y[1:, :] - H_Y[:-1, :])/DX + (H_X[:, :-1] - H_X[:, 1:])/DY)
    
    H_X[:, 1:-1] = H_X[:, 1:-1] + DT/(DY * mu_grid_HX[:, 1:-1]) * (E_Z[:, :-1] - E_Z[:, 1:])
    H_Y[1:-1, :] = H_Y[1:-1, :] + DT/(DX * mu_grid_HY[1:-1, :]) * (E_Z[1:, :] - E_Z[:-1, :])

    # Boundary conditions for the magnetic field
    # Half step of Ez
    E_Z_half = E_Z + 0.5*(DT / eps_grid) * ((H_Y[1:, :] - H_Y[:-1, :])/DX + (H_X[:, :-1] - H_X[:, 1:])/DY)

    H_X[:, 0] = H_X[:, -2] # Periodic
    H_X[:, -1] = H_X[:, 1]
    w = 2*np.pi * 4
    # Silver-Muller
    H_Y[0, :] = np.sqrt(eps_1/mu_1) * (E_Z_half[0, :] - plane_source_door(t=n*DT, tstart=0, tstop=2*Tp, w=w)[0]) + plane_source_door(t=n*DT, tstart=0, tstop=2*Tp, w=w)[1]
    H_Y[-1, :] = - np.sqrt(eps_1/mu_1) * E_Z_half[-1, :]

    
    # Boundary conditions ABC + PBC
    # Half step of Hy
    
    # E_Z[:, 0] = 0
    # E_Z[:, -1] = 0
    # E_Z[0, :] = np.sqrt(mu_r/eps_r) * (H_Y_half[0, :] - plane_wave(MESH_X[0, :], n*DT, 2*PI, 2*PI/Tp)[1]) + plane_wave(MESH_X[0, :], n*DT, 2*PI, 2*PI/Tp)[1]
    # E_Z[-1, :] = 0
    
    
    # # Analytical solution
    # E_Z_ref, _, _ = PEC_cavity(MESH_X, MESH_Y, (n+0.5)*DT)
    # _, H_X_ref, _ = PEC_cavity(MESH_HX_X, MESH_HX_Y, (n+1)*DT)
    # _, _, H_Y_ref = PEC_cavity(MESH_HY_X, MESH_HY_Y, (n+1)*DT)
    
    # E_Z_array[n] = E_Z
    # H_X_array[n] = H_X
    # H_Y_array[n] = H_Y

    # E_Z_ref_array[n] = E_Z_ref
    # H_X_ref_array[n] = H_X_ref
    # H_Y_ref_array[n] = H_Y_ref
    
    if n % 100 == 0:
        # fig = plt.figure()
        # ax = axes3d.Axes3D(fig)
        # ax.plot_wireframe(MESH_X, MESH_Y, E_Z_ref, rstride=1, cstride=1)
        # ax.set_zlim(-1, 1)
        # plt.savefig(f"2D_FDTD_snapshots/exact_snap_{n}.png")       
        # plt.close()
        
        # fig = plt.figure()
        # ax = axes3d.Axes3D(fig)
        # ax.plot_wireframe(MESH_X, MESH_Y, E_Z, rstride=1, cstride=1)
        # ax.set_zlim(-1, 1)
        # plt.savefig(f"2D_FDTD_snapshots/num_snap_{n}.png")       
        # plt.close()
        
        # plt.figure()
        # plt.pcolormesh(MESH_X, MESH_Y, E_Z_ref, vmin=-1, vmax=1, cmap='seismic')
        # plt.axis('scaled')
        # plt.colorbar()
        # plt.title(f'$t = {n*DT:.3f}$')
        # plt.tight_layout()
        # plt.savefig(f"FDTD_cavity/exact_snap_{n}.png")       
        # plt.close()
    
        plt.figure(figsize=(7*0.8,6*0.8))
        plt.pcolormesh(MESH_X, MESH_Y, E_Z, vmin=-1, vmax=1)
        plt.axis('scaled')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.colorbar()
        plt.title(f'$t = {n*DT:.3f}$')
        plt.tight_layout()
        plt.savefig(f"FDTD_ABC+PBC/snapshots/num_snap_{n}.png")       
        plt.close()

# # =============================================================================
# # Plot field maps at t = 0 and t = T/4
# # =============================================================================

# # t = 0

# fig = plt.figure(figsize=(12, 4.3))

# grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
#                  nrows_ncols=(1,3),
#                  axes_pad=0.4,
#                  share_all=True,
#                  cbar_location="right",
#                  cbar_mode="single",
#                  cbar_size="7%",
#                  cbar_pad=0.15,
#                  )

# # Add data to image grid
# grid[0].pcolormesh(MESH_HX_X, MESH_HX_Y, H_X_array[0], vmin=-1, vmax=1)
# grid[0].set_title(r'$H_x$')
# grid[0].set_ylabel(r'$y$')
# grid[1].pcolormesh(MESH_HY_X, MESH_HY_Y, H_Y_array[0], vmin=-1, vmax=1)
# grid[1].set_title(r'$H_y$')
# im = grid[2].pcolormesh(MESH_X, MESH_Y, E_Z_array[0], vmin=-1, vmax=1)
# grid[2].set_title(r'$E_z$')

# for ax in grid:
#     ax.axis('scaled')
#     ax.set_xlabel(r'$x$')

# # Colorbar
# ax.cax.colorbar(im)
# ax.cax.toggle_label(True)
# plt.suptitle('$t=0$')
# plt.tight_layout()
# plt.savefig(f'FDTD_cavity/PEC_field_map_t=0.png')


# # Plots at t = T/4
# t0 = Tp / 4
# n0 = int(t0/DT)
# fig = plt.figure(figsize=(12, 4.3))

# grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
#                  nrows_ncols=(1,3),
#                  axes_pad=0.4,
#                  share_all=True,
#                  cbar_location="right",
#                  cbar_mode="single",
#                  cbar_size="7%",
#                  cbar_pad=0.15,
#                  )

# # Add data to image grid
# grid[0].pcolormesh(MESH_HX_X, MESH_HX_Y, H_X_array[n0], vmin=-1, vmax=1)
# grid[0].set_title(r'$H_x$')
# grid[0].set_ylabel(r'$y$')
# grid[1].pcolormesh(MESH_HY_X, MESH_HY_Y, H_Y_array[n0], vmin=-1, vmax=1)
# grid[1].set_title(r'$H_y$')
# im = grid[2].pcolormesh(MESH_X, MESH_Y, E_Z_array[n0], vmin=-1, vmax=1)
# grid[2].set_title(r'$E_z$')

# for ax in grid:
#     ax.axis('scaled')
#     ax.set_xlabel(r'$x$')

# # Colorbar
# ax.cax.colorbar(im)
# ax.cax.toggle_label(True)
# plt.suptitle('$t=T/4$')
# plt.tight_layout()
# plt.savefig(f'FDTD_cavity/PEC_field_map_t=0.25T.png')

# # Plot temporal evolution of the analytical and numerical solution (Ez)

# t_grid = np.linspace(0, T/Tp, NT)

# fig, ax = plt.subplots(1, 1, figsize=(10, 4))

# ax.plot(t_grid, E_Z_ref_array[:, NX//2, NY//2], 'k-', label='exact')
# ax.plot(t_grid, E_Z_array[:, NX//2, NY//2], 'r--', label=f"FDTD ($N_x = {{{NX}}}$)")
# # ax.set_title(r'$N_x = N_y = {} \Rightarrow \Delta t = {:.3f}$'.format(NX, DT))

# ax.set_xlabel(r'Time (period)')
# ax.set_ylabel(r'$E_z(x=0.5, y=0.5) (t)$')
# ax.grid()
# plt.legend(loc='upper left')

# plt.tight_layout()

# #plt.legend(loc = 'upper left')
# plt.savefig(f'FDTD_cavity/comp_exact_num_Ez_Nx={NX}.png')


# #%%
# # ============================================
# # Plot temporal evolution of L2 relative error
# # ============================================

# t_grid = np.linspace(0, T/Tp, NT)
# eps = 1e-8

# L2_Hx = np.sqrt(np.mean((H_X_array - H_X_ref_array)**2, axis=(1,2))) #/ np.sqrt(np.mean((H_X_ref_array+eps)**2, axis=(1,2)))
# L2_Hy = np.sqrt(np.mean((H_Y_array - H_Y_ref_array)**2, axis=(1,2))) #/ np.sqrt(np.mean((H_Y_ref_array+eps)**2, axis=(1,2)))
# L2_Ez = np.sqrt(np.mean((E_Z_array - E_Z_ref_array)**2, axis=(1,2))) #/ np.sqrt(np.mean((E_Z_ref_array+eps)**2, axis=(1,2)))

# # np.savetxt(f'FDTD_cavity/L2_NX={NX}.txt', np.stack((t_grid, err), axis=1))
# plt.figure()
# plt.plot(t_grid, L2_Hx, 'b-', label='$L^2(H_x)$')
# plt.plot(t_grid, L2_Hy, 'r-', label='$L^2(H_y)$')
# plt.plot(t_grid, L2_Ez, 'k-', label='$L^2(E_z)$')

# #%%
# # NX_list = [51, 101, 151]
# # plt.figure()
# # for NX in NX_list:
# #     data = np.loadtxt(f'L2_error/L2_NX={NX}.txt')
# #     t_grid = data[:, 0]
# #     err = data[:, 1]
    
# #     plt.semilogy(t_grid, err, label=f'$N_x = {NX}$')

# plt.grid()
# plt.legend()
# plt.xlabel('Time (period)')
# plt.ylabel('relative $L^2$')
# plt.tight_layout()
# plt.savefig(f'FDTD_cavity/L2_error(t).png')

