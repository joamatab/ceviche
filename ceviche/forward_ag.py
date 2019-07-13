from autograd.core import make_vjp as _make_vjp, make_jvp as _make_jvp
from autograd.wrap_util import unary_to_nary
from autograd.extend import primitive, defvjp_argnum, vspace
from autograd.differential_operators import jacobian as jacobian_backward

import autograd.numpy as np

N = 2
A = np.random.random((N,N))

@unary_to_nary
def jacobian_backward(fun, x):
    """
    Returns a function which computes the Jacobian of `fun` with respect to
    positional argument number `argnum`, which must be a scalar or array. Unlike
    `grad` it is not restricted to scalar-output functions, but also it cannot
    take derivatives with respect to some argument types (like lists or dicts).
    If the input to `fun` has shape (in1, in2, ...) and the output has shape
    (out1, out2, ...) then the Jacobian has shape (out1, out2, ..., in1, in2, ...).
    """
    vjp, ans = _make_vjp(fun, x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return np.reshape(np.stack(grads), jacobian_shape)

# @unary_to_nary
def jacobian_forward(fun, x):
    """
    Returns a function which computes the Jacobian of `fun` with respect to
    positional argument number `argnum`, which must be a scalar or array. Unlike
    `grad` it is not restricted to scalar-output functions, but also it cannot
    take derivatives with respect to some argument types (like lists or dicts).
    If the input to `fun` has shape (in1, in2, ...) and the output has shape
    (out1, out2, ...) then the Jacobian has shape (out1, out2, ..., in1, in2, ...).
    """
    jvp, ans = _make_jvp(fun, x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(jvp, ans_vspace.standard_basis())
    return np.reshape(np.stack(grads), jacobian_shape)

def jacobian_forward(fun, x):
    jvp_f_x = _make_jvp(fun)(x)
    ans, jac_0 = 
    for v in ans


@unary_to_nary
def jacobian(fun, argnum, mode='reverse'):
    print(argnum)
    if mode == 'reverse':
        return jacobian_backward(fun, argnum)
    elif mode == 'forward':
        return jacobian_forward(fun, argnum)
    else:
        raise ValueError("'mode' kwarg must be one of {reverse, forward}")


def fn(x):
    return A @ x

if __name__ == '__main__':
    x0 = np.random.random((N,))
    # jacobian_backward(fn, 0)(x0)
    jacobian_forward(fn, x0)


