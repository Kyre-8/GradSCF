"""Static topology and differentiable Gaussian parameters."""
from dataclasses import dataclass
import jax
from jaxtyping import Array

@dataclass(frozen=True)
class BasisTopology:
    """Static shell structure. Values of trainable parameters are not stored here."""

    angular_momenta: tuple[int, ...]
    primitive_counts: tuple[int, ...]
    contraction_counts: tuple[int, ...]
    nuclear_charges: tuple[int, ...]
    cart: bool = True

    @property
    def nao(self):
        return sum(((l+1)*(l+2)//2 if self.cart else 2*l+1)*nc
                   for l, nc in zip(self.angular_momenta, self.contraction_counts))

    def __post_init__(self):
        n = len(self.angular_momenta)
        if not n or len(self.primitive_counts) != n or len(self.contraction_counts) != n:
            raise ValueError("Shell topology arrays must have the same nonzero length.")
        if any(l < 0 for l in self.angular_momenta):
            raise ValueError("Angular momenta must be nonnegative.")
        if any(n < 1 for n in self.primitive_counts + self.contraction_counts):
            raise ValueError("Shell primitive/contraction counts must be positive.")

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BasisParameters:
    """Dynamic raw basis values; coordinates in Bohr and exponents in Bohr^-2."""

    exponents: tuple[Array, ...]
    coefficients: tuple[Array, ...]
    centers: Array
    nuclear_coords: Array

    def tree_flatten(self):
        return (self.exponents, self.coefficients, self.centers, self.nuclear_coords), None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)


def check_parameter_shapes(topology, parameters):
    n = len(topology.angular_momenta)
    if len(parameters.exponents) != n or len(parameters.coefficients) != n:
        raise ValueError("Basis parameter shell counts do not match the topology.")
    if parameters.centers.shape != (n, 3):
        raise ValueError("Basis centers must have shape (nshell,3).")
    if parameters.nuclear_coords.shape != (len(topology.nuclear_charges), 3):
        raise ValueError("Nuclear coordinates do not match the topology.")
    for a, c, np_, nc in zip(parameters.exponents, parameters.coefficients,
                             topology.primitive_counts, topology.contraction_counts):
        if a.shape != (np_,) or c.shape != (np_, nc):
            raise ValueError("Exponent/coefficient shape does not match shell topology.")
