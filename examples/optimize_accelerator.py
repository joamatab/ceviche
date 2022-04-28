import unittest
import numpy as np
import matplotlib.pylab as plt
import autograd.numpy as npa

from scipy.optimize import minimize

import sys
sys.path.append('../ceviche')

from ceviche import fdfd_hz, jacobian
from ceviche.constants import *
from ceviche.utils import imarr, get_value

""" Optimize energy gain of dielectric particle accelerator """

# whether to plot the setup images
PLOT = False

# make parameters
wavelength = 2e-6                      # free space wavelength
omega = 2 * np.pi * C_0 / wavelength   # angular frequency
beta = .5                              # speed of electron / speed of light
dL = wavelength / 50.0                # grid size (m)

Nx, Ny = 450, int(beta * wavelength / dL)

eps_max = 3
eps_r = np.ones((Nx, Ny))
source = np.zeros((Nx, Ny))
source[30, :] = 10
source[-30-1, :] = -10
npml = [20, 0]
spc = 100
gap = 20

# make design region
design_region = np.zeros((Nx, Ny))
design_region[spc:Nx//2-gap//2, :] = 1
design_region[Nx//2+gap//2:Nx-spc, :] = 1
eps_r[design_region == 1] = eps_max

# make the accelration probe
eta = np.zeros((Nx, Ny), dtype=np.complex128)
channel_ys = np.arange(Ny)
eta[Nx//2, :] = np.exp(1j * 2 * np.pi * channel_ys / Ny)

# plot the probe through channel
if PLOT:
    plt.plot(np.real(imarr(eta[Nx//2,:])), label='RE\{eta\}')
    plt.xlabel('position along channel (y)')
    plt.ylabel('eta (y)')
    plt.show()

# vacuum test, get normalization
F = fdfd_hz(omega, dL, eps_r, npml)
Ex, Ey, Hz = F.solve(source)
E_mag = np.sqrt(np.square(np.abs(Ex)) + np.square(np.abs(Ey)))
E0 = np.max(E_mag[spc:-spc, :])
print(f'E0 = {E0} V/m')

# plot the vacuum fields
if PLOT:
    plt.imshow(np.real(Ey) / E0, cmap='RdBu')
    plt.title('E_y / E0 (<-)')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

# maximum electric field magnitude in the domain
def Emax(Ex, Ey, eps_r):
    E_mag = npa.sqrt(npa.square(npa.abs(Ex)) + npa.square(npa.abs(Ey)))
    material_density = (eps_r - 1) / (eps_max - 1)
    return npa.max(E_mag * material_density)

# average electric field magnitude in the domain
def Eavg(Ex, Ey):
    E_mag = npa.sqrt(npa.square(npa.abs(Ex)) + npa.square(npa.abs(Ey)))
    return npa.mean(E_mag)

# defines the acceleration gradient as a function of the relative permittivity grid
def accel_gradient(eps_arr):

    # set the permittivity of the FDFD and solve the fields
    F.eps_r = eps_arr.reshape((Nx, Ny))
    Ex, Ey, Hz = F.solve(source)

    # compute the gradient and normalize if you want
    G = npa.sum(Ey * eta / Ny) / Emax(Ex, Ey, eps_r)
    return -np.abs(G)

# define the gradient for autograd
grad_g = jacobian(accel_gradient)

# optimization
NIter = 200000
bounds_eps = [(1, eps_max) if design_region.flatten()[i] == 1 else (1,1) for i in range(eps_r.size)]
minimize(accel_gradient, eps_r.flatten(), args=(), method='L-BFGS-B', jac=grad_g,
    bounds=bounds_eps, tol=None, callback=None,
    options={'disp': True,
             'maxcor': 10,
             'ftol': 2.220446049250313e-09,
             'gtol': 1e-05,
             'eps': 1e-08,
             'maxfun': 15000,
             'maxiter': NIter,
             'iprint': -1,
             'maxls': 20})

def stack(arr, num_periods=1):
    # returns an array that is stacked `num_periods` times along y.  For plotting many periods of the accelerator unit cell.
    arr_orig = get_value(arr).copy()
    arr_big = get_value(arr).copy()
    for _ in range(num_periods):
        arr_big = np.hstack([arr_big, arr_orig])
    return arr_big

num_periods = 4

# plot the final permittivity
plt.imshow(imarr(stack(F.eps_r, num_periods=num_periods)), cmap='nipy_spectral')
plt.colorbar()
plt.show()

# plot the accelerating fields
Ex, Ey, Hz = F.solve(source)
plt.imshow(imarr(np.real(stack(Ey, num_periods=num_periods)))  / E0 , cmap='RdBu')
plt.title('E_y / E0 (<-)')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()
