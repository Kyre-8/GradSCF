"""Explicit backend contracts; availability and derivative support are separate."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    operators: tuple[str, ...]
    devices: tuple[str, ...]
    jit: bool
    derivative_variables: tuple[str, ...] = ()
    max_derivative_order: int = 0
    representations: tuple[str, ...] = ("cartesian",)
    layouts: tuple[str, ...] = ("full",)
    dtypes: tuple[str, ...] = ("float64",)
    ad_modes: tuple[str, ...] = ()

    def supports(self, operator, *, variable=None, derivative_order=0, layout='full'):
        if layout not in self.layouts:return False
        if layout!='full' and operator!='eri':return False
        if self.name=='native' and layout!='full' and derivative_order>0:return False
        if (self.name=='native' and variable in {'coefficients','exponents'} and derivative_order>0
                and operator not in {'overlap','kinetic','nuclear','dipole'}):return False
        if self.name=='native' and variable=='exponents' and derivative_order>1:return False
        if operator not in self.operators or derivative_order < 0:
            return False
        if operator == "ecp" and derivative_order > 0:
            return False
        if variable == "origin" and operator != "dipole" and derivative_order > 0:
            return False
        return derivative_order == 0 or (
            derivative_order <= self.max_derivative_order and variable in self.derivative_variables
        )


def backend_capabilities(name):
    operators = ("overlap", "kinetic", "nuclear", "dipole", "eri")
    if name == "native":
        return BackendCapabilities(name, operators+("ecp",), ("cpu",), True,
                                   ("centers", "nuclear_coords", "origin", "coefficients", "exponents"), 2,
                                   representations=("cartesian", "spherical"),
                                   layouts=("full", "s4", "s8"),
                                   ad_modes=("jvp", "vjp"))
    if name == "jax_reference":
        return BackendCapabilities(name, operators, ("cpu", "gpu"), True,
                                   ("centers", "nuclear_coords", "exponents", "coefficients"), 1,
                                   layouts=("full", "s4"), dtypes=("float32", "float64"),
                                   ad_modes=("jvp", "vjp"))
    raise ValueError(f"Unknown integral backend: {name!r}")
