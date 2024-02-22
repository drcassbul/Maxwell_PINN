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

def PEC_cavity(x, y, t):
    w = PI*sqrt(2)
    Ez = sin(PI*x) * sin(PI*y) * cos(w*t)
    Hx = -1/sqrt(2) * sin(PI*x) * cos(PI*y) * sin(w*t)
    Hy = 1/sqrt(2) * cos(PI*x) * sin(PI*y) * sin(w*t)
    
    return Ez, Hx, Hy




# =============================================================================
# Physical constants
# =============================================================================

PERMITTIVITY_VACUUM = c.epsilon_0 # Electric permittivity of vacuum
PERMEABILITY_VACUUM = c.mu_0 # Permeability of vacuum

# =============================================================================
# Numerical parameters
# =============================================================================

# Dimensions of grid
LX = 1
LY = 1

# Number of grid points
NX = 201
NY = 201

# Space increments
DX = LX / (NX - 1)
DY = LY / (NY - 1)

# Time increment (CFL condition)
DT = 1 / sqrt(1/DX**2 + 1/DY**2)

# Total integration time
Tp = sqrt(2) # Time period
T = 1 * Tp
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

E_Z, _, _ = PEC_cavity(MESH_X, MESH_Y, -0.5*DT)
_, H_X, _ = PEC_cavity(MESH_HX_X, MESH_HX_Y, 0)
_, _, H_Y = PEC_cavity(MESH_HY_X, MESH_HY_Y, 0)


# fig = plt.figure()
# ax = axes3d.Axes3D(fig)
# ax.plot_wireframe(MESH_X, MESH_Y, E_Z, rstride=2, cstride=2)

# =============================================================================
# Yee's algorithm
# =============================================================================

E_Z_array = np.full((NT, *E_Z.shape), np.nan)
H_X_array = np.full((NT, *H_X.shape), np.nan)
H_Y_array = np.full((NT, *H_Y.shape), np.nan)

E_Z_ref_array = np.full((NT, *E_Z.shape), np.nan)
H_X_ref_array = np.full((NT, *H_X.shape), np.nan)
H_Y_ref_array = np.full((NT, *H_Y.shape), np.nan)

for n in range(NT):
    E_Z = E_Z + DT * ((H_Y[1:, :] - H_Y[:-1, :])/DX + (H_X[:, :-1] - H_X[:, 1:])/DY)
    
    H_X[:, 1:-1] = H_X[:, 1:-1] + DT/DY * (E_Z[:, :-1] - E_Z[:, 1:])
    H_Y[1:-1, :] = H_Y[1:-1, :] + DT/DX * (E_Z[1:, :] - E_Z[:-1, :])

    # Boundary conditions for the magnetic field
    H_X[:, 0] = H_X[:, 1]
    H_X[:, -1] = H_X[:, -2]
    H_Y[0, :] = H_Y[1, :]
    H_Y[-1, :] = H_Y[-2, :]

    # Boundary conditions for the electric field (PEC)
    E_Z[:, 0] = 0
    E_Z[:, -1] = 0
    E_Z[0, :] = 0
    E_Z[-1, :] = 0
    
    
    # Analytical solution
    E_Z_ref, _, _ = PEC_cavity(MESH_X, MESH_Y, (n+0.5)*DT)
    _, H_X_ref, _ = PEC_cavity(MESH_HX_X, MESH_HX_Y, (n+1)*DT)
    _, _, H_Y_ref = PEC_cavity(MESH_HY_X, MESH_HY_Y, (n+1)*DT)
    
    E_Z_array[n] = E_Z
    H_X_array[n] = H_X
    H_Y_array[n] = H_Y

    E_Z_ref_array[n] = E_Z_ref
    H_X_ref_array[n] = H_X_ref
    H_Y_ref_array[n] = H_Y_ref
    
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
    
        plt.figure()
        plt.pcolormesh(MESH_X, MESH_Y, E_Z, vmin=-1, vmax=1)
        plt.axis('scaled')
        plt.colorbar()
        plt.title(f'$t = {n*DT:.3f}$')
        plt.tight_layout()
        plt.savefig(f"FDTD_cavity/snapshots/num_snap_{n}.png")       
        plt.close()

# =============================================================================
# Plot field maps at t = 0 and t = T/4
# =============================================================================

# t = 0
        
        

fig = plt.figure(figsize=(12, 4.3))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.4,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )

# Add data to image grid
grid[0].pcolormesh(MESH_HX_X, MESH_HX_Y, H_X_array[0], vmin=-1, vmax=1)
grid[0].set_title(r'$H_x$')
grid[0].set_ylabel(r'$y$')
grid[1].pcolormesh(MESH_HY_X, MESH_HY_Y, H_Y_array[0], vmin=-1, vmax=1)
grid[1].set_title(r'$H_y$')
im = grid[2].pcolormesh(MESH_X, MESH_Y, E_Z_array[0], vmin=-1, vmax=1)
grid[2].set_title(r'$E_z$')

for ax in grid:
    ax.axis('scaled')
    ax.set_xlabel(r'$x$')

# Colorbar
ax.cax.colorbar(im)
ax.cax.toggle_label(True)
plt.suptitle('$t=0$')
plt.tight_layout()
plt.savefig(f'FDTD_cavity/PEC_field_map_t=0.png')


# Plots at t = T/4
t0 = Tp / 4
n0 = int(t0/DT)
fig = plt.figure(figsize=(12, 4.3))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.4,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )

# Add data to image grid
grid[0].pcolormesh(MESH_HX_X, MESH_HX_Y, H_X_array[n0], vmin=-1, vmax=1)
grid[0].set_title(r'$H_x$')
grid[0].set_ylabel(r'$y$')
grid[1].pcolormesh(MESH_HY_X, MESH_HY_Y, H_Y_array[n0], vmin=-1, vmax=1)
grid[1].set_title(r'$H_y$')
im = grid[2].pcolormesh(MESH_X, MESH_Y, E_Z_array[n0], vmin=-1, vmax=1)
grid[2].set_title(r'$E_z$')

for ax in grid:
    ax.axis('scaled')
    ax.set_xlabel(r'$x$')

# Colorbar
ax.cax.colorbar(im)
ax.cax.toggle_label(True)
plt.suptitle('$t=T/4$')
plt.tight_layout()
plt.savefig(f'FDTD_cavity/PEC_field_map_t=0.25T.png')

# Plot temporal evolution of the analytical and numerical solution (Ez)

t_grid = np.linspace(0, T/Tp, NT)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))

ax.plot(t_grid, E_Z_ref_array[:, NX//2, NY//2], 'k-', label='exact')
ax.plot(t_grid, E_Z_array[:, NX//2, NY//2], 'r--', label=f"FDTD ($N_x = {{{NX}}}$)")
# ax.set_title(r'$N_x = N_y = {} \Rightarrow \Delta t = {:.3f}$'.format(NX, DT))

ax.set_xlabel(r'Time (period)')
ax.set_ylabel(r'$E_z(x=0.5, y=0.5) (t)$')
ax.grid()
plt.legend(loc='upper left')

plt.tight_layout()

#plt.legend(loc = 'upper left')
plt.savefig(f'FDTD_cavity/comp_exact_num_Ez_Nx={NX}.png')


#%%
# ============================================
# Plot temporal evolution of L2 relative error
# ============================================

t_grid = np.linspace(0, T/Tp, NT)
eps = 1e-8

L2_Hx = np.sqrt(np.mean((H_X_array - H_X_ref_array)**2, axis=(1,2))) #/ np.sqrt(np.mean((H_X_ref_array+eps)**2, axis=(1,2)))
L2_Hy = np.sqrt(np.mean((H_Y_array - H_Y_ref_array)**2, axis=(1,2))) #/ np.sqrt(np.mean((H_Y_ref_array+eps)**2, axis=(1,2)))
L2_Ez = np.sqrt(np.mean((E_Z_array - E_Z_ref_array)**2, axis=(1,2))) #/ np.sqrt(np.mean((E_Z_ref_array+eps)**2, axis=(1,2)))

# np.savetxt(f'FDTD_cavity/L2_NX={NX}.txt', np.stack((t_grid, err), axis=1))
plt.figure()
plt.plot(t_grid, L2_Hx, 'b-', label='$L^2(H_x)$')
plt.plot(t_grid, L2_Hy, 'r-', label='$L^2(H_y)$')
plt.plot(t_grid, L2_Ez, 'k-', label='$L^2(E_z)$')

#%%
# NX_list = [51, 101, 151]
# plt.figure()
# for NX in NX_list:
#     data = np.loadtxt(f'L2_error/L2_NX={NX}.txt')
#     t_grid = data[:, 0]
#     err = data[:, 1]
    
#     plt.semilogy(t_grid, err, label=f'$N_x = {NX}$')

plt.grid()
plt.legend()
plt.xlabel('Time (period)')
plt.ylabel('relative $L^2$')
plt.tight_layout()
plt.savefig(f'FDTD_cavity/L2_error(t).png')

# #%%

# # =============================================================================
# # Plot temporal evolution of the analytical and numerical solution (Ez, Hx, Hy)
# # =============================================================================
# t_grid = np.linspace(0, T/Tp, NT)

# fig, ax = plt.subplots(3, 1, figsize=(10, 8))

# ax[0].plot(t_grid, E_Z_ref_array[:, NX//2, NY//2], 'k-', label='exact')
# ax[0].plot(t_grid, E_Z_array[:, NX//2, NY//2], 'r--', label=f"Yee ($\Delta x = {{{DX:.2f}}}$)")
# ax[0].set_title('$E_z(x=0.5, y=0.5) (t)$')

# ax[1].plot(t_grid, H_X_ref_array[:, NX//2, 0], 'k-', label='exact')
# ax[1].plot(t_grid, H_X_array[:, NX//2, 0], 'r--', label=f"Yee ($\Delta x = {{{DX:.2f}}}$)")
# ax[1].set_title('$H_x(x=0.5, y=0) (t)$')

# ax[2].plot(t_grid, H_Y_ref_array[:, 0, NX//2], 'k-', label='exact')
# ax[2].plot(t_grid, H_Y_array[:, 0, NX//2], 'r--', label=f"Yee ($\Delta x = {{{DX:.2f}}}$)")
# ax[2].set_title('$H_y(x=0, y=0.5) (t)$')

# ax[2].set_xlabel(r'Time (period)')
# plt.suptitle(r'$N_\textrm{{cells}} = {} \Rightarrow \Delta t = {:.3f}$'.format(NX-1, DT))
# plt.tight_layout()


# #plt.legend(loc = 'upper left')
# plt.savefig(f'PEC_cavity_figures/comp_exact_num_Nx={NX}.png')

# #%%
# t_grid = np.linspace(0, T/Tp, NT)

# fig, ax = plt.subplots(1, 1, figsize=(10, 2.8))

# ax.plot(t_grid, E_Z_ref_array[:, NX//2, NY//2], 'k-', label='exact')
# ax.plot(t_grid, E_Z_array[:, NX//2, NY//2], 'r--', label=f"Yee ($\Delta x = {{{DX:.2f}}}$)")
# ax.set_title(r'$N_\textrm{{cells}} = {} \Rightarrow \Delta t = {:.3f}$'.format(NX-1, DT))

# ax.set_xlabel(r'Time (period)')

# plt.tight_layout()

# #plt.legend(loc = 'upper left')
# plt.savefig(f'PEC_cavity_figures/comp_exact_num_Ez_Nx={NX}.png')

# #%%
# t_grid = np.linspace(0, T/Tp, NT)
# E_num = (np.sum(E_Z_array**2, axis=(1,2)) + np.sum(H_X_array**2, axis=(1,2)) + np.sum(H_Y_array**2, axis=(1,2)))/NX**2

# # =============================================================================
# # Save energy(t)
# # =============================================================================
# np.savetxt(f'energy/energy_NX={NX}.txt', np.stack((t_grid, E_num), axis=1))
# #%%
# # =============================================================================
# # Temporal evolution of energy
# # =============================================================================

# t_grid = np.linspace(0, T/Tp, NT)
# E_num = (np.sum(E_Z_array**2, axis=(1,2)) + np.sum(H_X_array**2, axis=(1,2)) + np.sum(H_Y_array**2, axis=(1,2)))/NX**2
# E_ref = (np.sum(E_Z_ref_array**2, axis=(1,2)) + np.sum(H_X_ref_array**2, axis=(1,2)) + np.sum(H_Y_ref_array**2, axis=(1,2)))/NX**2

# plt.figure()
# plt.plot(t_grid, E_num, 'k-', label='num.')
# plt.hlines(0.25, t_grid.min(), t_grid.max(), colors='k', linestyles='dashed')
# #plt.plot(t_grid, E_ref, label='exact')
# plt.legend()

# NX_list = [51, 101, 151]
# plt.figure()

# plt.hlines(0.25, t_grid.min(), t_grid.max(), colors='k', linestyles='dashed', label='theo.')
# for NX in NX_list:
#     data = np.loadtxt(f'energy/energy_NX={NX}.txt')
#     t_grid = data[:, 0]
#     E_num = data[:, 1]
    
#     plt.semilogy(t_grid, E_num, label=f'$N_x = {NX}$')


# plt.xlabel('Time (period)')
# plt.ylabel('Total energy')
# plt.xlim(0,5)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'energy/energy(t).png')


