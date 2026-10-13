import jax
import jax.numpy as jnp
import numpy as np


def test_smooth_matrix_function_degenerate_derivative():
    from gradscf.solvers.eigen.spectral import matrix_function
    a = jnp.eye(2)*.4
    b = jnp.array([[.2,.3],[.3,-.1]])
    function = lambda x,p:jnp.exp(p*x)
    f = lambda a,p:matrix_function(a,p,function)
    out, tangent = jax.jvp(f, (a,jnp.array(.7)), (b,jnp.array(.2)))
    np.testing.assert_allclose(out, jnp.eye(2)*jnp.exp(.28), atol=1e-12)
    np.testing.assert_allclose(tangent, jnp.exp(.28)*(.7*b+.08*jnp.eye(2)), atol=1e-12)
