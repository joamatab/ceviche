import unittest

import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import copy
import matplotlib.pylab as plt

from numpy.linalg import norm

from autograd.extend import primitive, defvjp
from autograd import grad

import sys
sys.path.append('../ceviche')

from ceviche.utils import grad_num, get_value, imarr
from ceviche.fdfd import fdfd_hz, fdfd_ez, fdfd_ez_nl
from ceviche.jacobians import jacobian
from ceviche.constants import *
"""
This file tests the autograd gradients of an FDFD and makes sure that they
equal the numerical derivatives
"""

# test parameters
ALLOWED_RATIO = 1e-4    # maximum allowed ratio of || grad_num - grad_auto || vs. || grad_num ||
VERBOSE = False         # print out full gradients?
DEPS = 1e-6             # numerical gradient step size

print("Testing the Nonlinear FDFD gradients")

class TestFDFD(unittest.TestCase):

    """ Tests the flexible objective function specifier """

    def setUp(self):

        # basic simulation parameters
        self.Nx = 20
        self.Ny = 20
        self.omega = 2*np.pi*200e12
        self.dL = 5e-8
        self.pml = [5, 5]

        # sources (chosen to have objectives around 1)
        self.source_amp_ez = 1e3
        self.source_amp_hz = 1e3

        self.source_ez = np.zeros((self.Nx, self.Ny))
        self.source_ez[self.Nx//2, self.Ny//2] = self.source_amp_ez

        self.source_hz = np.zeros((self.Nx, self.Ny))
        self.source_hz[self.Nx//2, self.Ny//2] = self.source_amp_hz

        # starting relative permittivity (random for debugging)
        self.eps_lin = np.random.random((self.Nx, self.Ny)) + 1    
        self.chi3 = 2000
        self.eps_nl = lambda Ez: self.eps_lin + 3 * self.chi3 * np.square(np.abs(Ez))

    def check_gradient_error(self, grad_num, grad_auto):
        """ Checks the test case:
            compares the norm of the gradient to the norm of the difference
            Throws error if this is greater than ALLOWED RATIO
        """
        norm_grad = norm(grad_num)
        print('\t\tnorm of gradient:   ', norm_grad)
        norm_diff = norm(grad_num - grad_auto)
        print('\t\tnorm of difference: ', norm_diff)
        norm_ratio = norm_diff / norm_grad        
        print('\t\tratio of norms:     ', norm_ratio)
        self.assertLessEqual(norm_ratio, ALLOWED_RATIO)
        print('')

    def test_Ez_lin(self):

        print('\ttesting linear Ez in FDFD')

        f = fdfd_ez(self.omega, self.dL, self.eps_lin, self.pml)

        def J_fdfd(eps_arr):

            f.eps_r = eps_arr.reshape((self.Nx, self.Ny))

            # set the source amplitude to the permittivity at that point
            Hx, Hy, Ez = f.solve(self.source_ez)

            return npa.sum(npa.square(npa.abs(Ez))) \
                 + npa.sum(npa.square(npa.abs(Hx))) \
                 + npa.sum(npa.square(npa.abs(Hy)))

        grad_autograd_rev = jacobian(J_fdfd, mode='reverse')(self.eps_lin)
        grad_numerical = jacobian(J_fdfd, mode='numerical')(self.eps_lin)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(self.eps_lin))
            print('\tgrad (auto):  \n\t\t', grad_autograd_rev)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        self.check_gradient_error(grad_numerical, grad_autograd_rev)


    def test_Ez_nl(self):

        print('\ttesting reverse-mode nonlinear Ez in FDFD')

        f = fdfd_ez_nl(self.omega, self.dL, self.eps_nl, self.pml)

        Hx, Hy, Ez = f.solve(self.source_ez)        
        mod_strength = self.chi3 * np.max(np.square(np.abs(Ez)))
        print('\t\tmodulation strength (chi3 * max(|E|^2)) = {}'.format(mod_strength))

        def J_fdfd(eps_arr):

            eps_lin = eps_arr.reshape((self.Nx, self.Ny))

            # construct nonlinear epsilon using autograd numpy wrapper
            eps_nl = lambda Ez: eps_lin + 3 * eps_lin * self.chi3 * npa.square(npa.abs(Ez))

            # set the permittivity
            f.eps_r = eps_nl

            # set the source amplitude to the permittivity at that point
            Hx, Hy, Ez = f.solve(self.source_ez)

            return npa.sum(npa.square(npa.abs(Ez))) \
                 + npa.sum(npa.square(npa.abs(Hx))) \
                 + npa.sum(npa.square(npa.abs(Hy)))

        grad_autograd_rev = jacobian(J_fdfd, mode='reverse')(self.eps_lin)
        grad_numerical = jacobian(J_fdfd, mode='numerical')(self.eps_lin)

        if VERBOSE:
            print('\tobjective function value: ', J_fdfd(self.eps_lin))
            print('\tgrad (auto):  \n\t\t', grad_autograd_rev)
            print('\tgrad (num):   \n\t\t', grad_numerical)

        self.check_gradient_error(grad_numerical, grad_autograd_rev)


if __name__ == '__main__':
    unittest.main()
