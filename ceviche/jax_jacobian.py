from jax import jacfwd, jacrev

def jacobian(fn, mode='reverse'):
	""" Switches between jax jacobian functions and does error checking """
	if mode == 'reverse':
		try:
			jac = jacrev(fn)
		except:
			raise ValueError("Could not compile jacobian for fn: {}, \
				make sure that you use jax.numpy to define the operations within".format(fn))
	elif mode == 'forward':
		try:
			jac = jacfwd(fn)
		except:
			raise ValueError("Could not compile jacobian for fn: {}, \
				make sure that you use jax.numpy to define the operations within".format(fn))
	else:
		raise ValueError("'mode' kwarg must be either 'forward' or 'reverse', given {}".format(mode))
	return jac


if __name__ == '__main__':

	import numpy as np

	N = 2
	A = np.random.random((N,N))
	print("A = \n", A)
	
	def fn(x):
		return A @ x

	x0 = np.random.random((N,))
	jac_for = jacobian(fn, mode='forward')
	jac_rev = jacobian(fn, mode='reverse')
	print("fn' = \n", jac_for(x0))
	print("fn' = \n", jac_rev(x0))
