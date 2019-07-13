from autograd.core import make_vjp, make_jvp
from autograd.wrap_util import unary_to_nary
from autograd.extend import vspace

import autograd.numpy as np

@unary_to_nary
def jacobian_reverse(fun, x):
    """ Compute jacobian of fun with respect to x using reverse mode differentiation"""
    vjp, ans = make_vjp(fun, x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return np.reshape(np.stack(grads), jacobian_shape)

@unary_to_nary
def jacobian_forward(fun, x):
    """ Compute jacobian of fun with respect to x using forward mode differentiation"""
    print(x)
    jvp = make_jvp(fun, x)
    ans = fun(x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(lambda b: jvp(b)[1], ans_vspace.standard_basis())
    return np.reshape(np.stack(grads), jacobian_shape).T

def jacobian(fun, argnum=0, mode='reverse'):
    """ Computes jacobian of `fun` with respect to argument number `argnum` using automatic differentiation """
    if mode == 'reverse':
        return jacobian_reverse(fun, argnum)
    elif mode == 'forward':
        return jacobian_forward(fun, argnum)
    else:
        raise ValueError("'mode' kwarg must be either 'reverse' or 'forward', given {}".format(mode))


if __name__ == '__main__':

    N = 2
    A = np.random.random((N,N))
    print('A = \n', A)

    def fn(x, b):
        return A @ x + A.T @ b

    x0 = np.random.random((N,))
    b0 = np.random.random((N,))    
    print('Jac_rev = \n', jacobian(fn, argnum=0, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=0, mode='forward')(x0, b0))

    print('A^T = \n', A.T)
    print('Jac_rev = \n', jacobian(fn, argnum=1, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=1, mode='forward')(x0, b0))
