#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:44:48 2022

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

def PEC_interface(x, y, t, a1, a2, b, w):
    Ez = np.full_like(x, np.nan)
    Hx = np.full_like(x, np.nan)
    Hy = np.full_like(x, np.nan)
    
    i_inter = np.argmin(np.abs(x[:,0] - 0.5))
    
    for i in range(i_inter):
        Ez[i] = sin(a1*x[i]) * sin(w*t) * sin(b*y[i])
        Hx[i] = b/w * sin(a1*x[i]) * cos(w*t) * cos(b*y[i])
        Hy[i] = -a1/w * cos(a1*x[i]) * cos(w*t) * sin(b*y[i])
        
    for i in range(i_inter, Ez.shape[0]):
        Ez[i] = cos(a2*x[i]) * sin(w*t) * sin(b*y[i])
        Hx[i] = b/w * cos(a2*x[i]) * cos(w*t) * cos(b*y[i])
        Hy[i] = a2/w * sin(a2*x[i]) * cos(w*t) * sin(b*y[i])
    return Ez, Hx, Hy

NX = 51
NY = 51

Tp = 2/sqrt(5)

# Initialize spatial grid in physical units
GRID_X = np.linspace(0, 1.25, NX)
GRID_Y = np.linspace(0, 1, NY)

# Staggered grid for Ez (NX-1 x NY-1)
MESH_X, MESH_Y = np.meshgrid(GRID_X, GRID_Y, indexing='ij')

t  = 0.75
Ez, Hx, Hy = PEC_interface(MESH_X, MESH_Y, t=t, a1=3*PI, a2=2*PI, b=PI, w=sqrt(5)*PI)

plt.figure()
plt.pcolormesh(MESH_X, MESH_Y, Ez, cmap='seismic')
plt.axis('scaled')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$E_z$')
plt.colorbar()
plt.tight_layout()
plt.savefig('PEC_cavity_interface_figures/PEC_interface_Ez.png')

plt.figure()
plt.pcolormesh(MESH_X, MESH_Y, Hx, cmap='seismic')
plt.axis('scaled')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$H_x$')
plt.colorbar()
plt.tight_layout()
plt.savefig('PEC_cavity_interface_figures/PEC_interface_Hx.png')

plt.figure()
plt.pcolormesh(MESH_X, MESH_Y, Hy, cmap='seismic')
plt.axis('scaled')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$H_y$')
plt.colorbar()
plt.tight_layout()
plt.savefig('PEC_cavity_interface_figures/PEC_interface_Hy.png')


fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))



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

#%%
y0 = 0.75
idx_y = np.argmin(np.abs(GRID_Y - 0.75))
plt.figure()
plt.plot(GRID_X, Ez[:, idx_y], 'k--', label=r'$E_z$')
plt.plot(GRID_X, Hx[:, idx_y], 'b-', label=r'$H_x$')
plt.plot(GRID_X, Hy[:, idx_y], 'r-', label=r'$H_y$')
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel('field components')
plt.title(r'$y=3/4$, $t=0.75$')
plt.ylim(-0.7, 0.7)
plt.vlines(0.5, -0.7, 0.7, colors=['k'])
plt.tight_layout()
plt.savefig('PEC_cavity_interface_figures/PEC_interface_fields_x.png')

#%%
# =============================================================================
# Numerical parameters
# =============================================================================

# Dimensions of grid
LX = 1.25
LY = 1

# Number of grid points
NX = 151
NY = 151

# Space increments
DX = LX / (NX - 1)
DY = LY / (NY - 1)

# Time increment (CFL condition)
DT = 1/sqrt(1/DX**2 + 1/DY**2) / sqrt(2)

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

FULL_MESH_X, FULL_MESH_Y = np.meshgrid(np.linspace(0, (NX-1)*DX, 2*NX-1), np.linspace(0, (NY-1)*DY, 2*NY-1), indexing='ij')


PERMITTIVITY = np.array([1, 2])
PERMEABILITY = np.array([1, 1])
ELECTRIC_CONDUCTIVITY = np.array([0, 0])
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

MEDIA = np.where((FULL_MESH_X <= 0.5), 1, 0)
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
    
# =============================================================================
# Initial conditions
# =============================================================================

E_Z, _, _ = PEC_interface(MESH_X, MESH_Y, t=-0.5*DT, a1=3*PI, a2=2*PI, b=PI, w=sqrt(5)*PI)
_, H_X, _ = PEC_interface(MESH_HX_X, MESH_HX_Y, t=0, a1=3*PI, a2=2*PI, b=PI, w=sqrt(5)*PI)
_, _, H_Y = PEC_interface(MESH_HY_X, MESH_HY_Y, t=0, a1=3*PI, a2=2*PI, b=PI, w=sqrt(5)*PI)


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
    E_Z = C_A_grid[1::2, 1::2] * E_Z + C_Bx_grid[1::2, 1::2] * (H_Y[1:, :] - H_Y[:-1, :]) + C_By_grid[1::2, 1::2] * (H_X[:, :-1] - H_X[:, 1:])
    
    H_X[:, 1:-1] = D_A_grid[1::2, 0::2][:, 1:-1] * H_X[:, 1:-1] + D_By_grid[1::2, 0::2][:, 1:-1] * (E_Z[:, :-1] - E_Z[:, 1:])
    H_Y[1:-1, :] = D_A_grid[0::2, 1::2][1:-1, :] * H_Y[1:-1, :] + D_Bx_grid[0::2, 1::2][1:-1, :] * (E_Z[1:, :] - E_Z[:-1, :])

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
    E_Z_ref, _, _ = PEC_interface(MESH_X, MESH_Y, t=(n+0.5)*DT, a1=3*PI, a2=2*PI, b=PI, w=sqrt(5)*PI)
    _, H_X_ref, _ = PEC_interface(MESH_HX_X, MESH_HX_Y, t=(n+1)*DT, a1=3*PI, a2=2*PI, b=PI, w=sqrt(5)*PI)
    _, _, H_Y_ref = PEC_interface(MESH_HY_X, MESH_HY_Y, t=(n+1)*DT, a1=3*PI, a2=2*PI, b=PI, w=sqrt(5)*PI)
    
    E_Z_array[n] = E_Z
    H_X_array[n] = H_X
    H_Y_array[n] = H_Y

    E_Z_ref_array[n] = E_Z_ref
    H_X_ref_array[n] = H_X_ref
    H_Y_ref_array[n] = H_Y_ref
    
    if n % 1000000 == 0:
        plt.figure()
        plt.pcolormesh(MESH_X, MESH_Y, E_Z_ref, vmin=-1, vmax=1, cmap='seismic')
        plt.axis('scaled')
        plt.colorbar()
        plt.title(f'$t={(n+1)*DT}$')
        plt.savefig(f"PEC_cavity_interface_figures/snapshots/exact_snap_{n}.png")       
        plt.close()
    
        plt.figure()
        plt.pcolormesh(MESH_X, MESH_Y, E_Z, vmin=-1, vmax=1, cmap='seismic')
        plt.axis('scaled')
        plt.colorbar()
        plt.savefig(f"PEC_cavity_interface_figures/snapshots/num_snap_{n}.png")       
        plt.close()
        
        
#%%
# =============================================================================
# Plot temporal evolution of euclidian error (L2 error)
# =============================================================================

t_grid = np.linspace(0, T/Tp, NT)

err = 1/((sqrt(NX)*sqrt(NY))) * (np.sum((E_Z_array - E_Z_ref_array)**2, axis=(1,2)) + np.sum((H_X_array - H_X_ref_array)**2, axis=(1,2)) + np.sum((H_Y_array - H_Y_ref_array)**2, axis=(1,2)))**(0.5) 
np.savetxt(f'PEC_cavity_interface_figures/L2_error/L2_NX={NX}.txt', np.stack((t_grid, err), axis=1))
plt.figure()
plt.semilogy(t_grid, err)

#%%
NX_list = [51, 101, 151]
plt.figure()
for NX in NX_list:
    data = np.loadtxt(f'PEC_cavity_interface_figures/L2_error/L2_NX={NX}.txt')
    t_grid = data[:, 0]
    err = data[:, 1]
    
    plt.semilogy(t_grid, err, label=f'$N_x = {NX}$')

plt.grid()
plt.legend()
plt.xlabel('Time (period)')
plt.ylabel('$L^2$ error')
plt.tight_layout()
plt.savefig(f'PEC_cavity_interface_figures/L2_error/L2_error_interface(t).png')

#%%
t_grid = np.linspace(0, T/Tp, NT)
E_num = (np.sum(E_Z_array**2, axis=(1,2)) + np.sum(H_X_array**2, axis=(1,2)) + np.sum(H_Y_array**2, axis=(1,2)))/NX**2

# =============================================================================
# Save energy(t)
# =============================================================================
np.savetxt(f'PEC_cavity_interface_figures/energy/energy_interface_NX={NX}.txt', np.stack((t_grid, E_num), axis=1))
#%%
# =============================================================================
# Temporal evolution of energy
# =============================================================================

t_grid = np.linspace(0, T/Tp, NT)
E_num = (np.sum(E_Z_array**2, axis=(1,2)) + np.sum(H_X_array**2, axis=(1,2)) + np.sum(H_Y_array**2, axis=(1,2)))/NX**2
E_ref = (np.sum(E_Z_ref_array**2, axis=(1,2)) + np.sum(H_X_ref_array**2, axis=(1,2)) + np.sum(H_Y_ref_array**2, axis=(1,2)))/NX**2

plt.figure()
plt.plot(t_grid, E_num, 'k-', label='num.')
plt.hlines(0.25, t_grid.min(), t_grid.max(), colors='k', linestyles='dashed')
#plt.plot(t_grid, E_ref, label='exact')
plt.legend()

NX_list = [51, 101, 151]
plt.figure()

plt.hlines(0.25, t_grid.min(), t_grid.max(), colors='k', linestyles='dashed', label='theo.')
for NX in NX_list:
    data = np.loadtxt(f'PEC_cavity_interface_figures/energy/energy_interface_NX={NX}.txt')
    t_grid = data[:, 0]
    E_num = data[:, 1]
    
    plt.semilogy(t_grid, E_num, label=f'$N_x = {NX}$')


plt.xlabel('Time (period)')
plt.ylabel('Total energy')
plt.xlim(0,5)
plt.legend()
plt.tight_layout()
plt.savefig(f'PEC_cavity_interface_figures/energy/energy_interface(t).png')