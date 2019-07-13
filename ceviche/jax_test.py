from jax import jacfwd, jacrev

import jax.numpy as np
import numpy as npo

N = 2
A = npo.random.random((N,N))
print(A)
def fn(x):
    return A @ x

if __name__ == '__main__':
    x0 = npo.random.random((N,))
    # jacobian_backward(fn, 0)(x0)
    print(jacfwd(fn)(x0))
    print(jacrev(fn)(x0))
