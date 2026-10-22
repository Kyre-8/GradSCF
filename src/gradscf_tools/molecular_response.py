"""A molecular energy E(R,F) for small, fully relaxed derivative examples.

All electronic solves use GradSCF's shared orbital optimizer and implicit
response. This adapter only assembles moving molecular inputs and the electric
field interaction. It contains no SCF, response or diagonalization algorithm.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals, scf
from gradscf.dft.xc import xc_type
from gradscf.scf.inputs import build_rks_integral_inputs, build_uks_integral_inputs
from gradscf.scf.rsh import local_xc_spec, rsh_preset


class MolecularEnergy:
    """Relaxed real HF/LDA/GGA/hybrid E(R,F) in atomic units.

    Construct from converged RHF/RKS or UHF/UKS with Cartesian AOs and integer
    occupations. Restricted open-shell and noncollinear references are excluded.
    Coordinates are in bohr, F in atomic electric-field units, E in Hartree.
    E(F) = E(0) - mu.F - F.alpha.F/2 + ... includes the nuclear field energy.
    The initial orbitals seed each forward solve; they are not a frozen density.
    Every call transports the AO metric and relaxes the orbitals again. The
    implicit backward follows this stationary branch, including in nested AD.

    Full native ERIs are the default geometry-AD representation. HF may opt
    into ``jk_backend="direct"`` to use moving native J/K without AO ERIs.
    Direct calls inherit the seed's J/K worker count and retain zero screening.
    The default requires a full-J/K seed; explicit direct accepts full or
    direct HF seeds. Original omegaB97X-D carries full/LR ERIs and its
    geometry-dependent CHG energy explicitly through the same solver.
    DFT grids and AO values move with R. Isolated state stability and numerical
    grid convergence must be checked for each demonstrated observable.
    """

    def __init__(self, mf, *, mode="implicit", gradient_tolerance=1e-9,
                 jk_backend="full"):
        if jk_backend not in ("full", "direct"):
            raise ValueError("jk_backend must be 'full' or 'direct'.")
        if not jax.config.x64_enabled:
            raise ValueError("MolecularEnergy requires JAX float64.")
        if not bool(mf.converged):
            raise ValueError("Supply a converged molecular reference.")
        coefficients = np.asarray(mf.mo_coeff)
        occupations = np.asarray(mf.mo_occ)
        if np.iscomplexobj(coefficients) or coefficients.ndim not in (2, 3):
            raise ValueError("MolecularEnergy requires real restricted or unrestricted orbitals.")
        self.unrestricted = coefficients.ndim == 3
        if self.unrestricted:
            if (coefficients.shape[0] != 2 or occupations.shape != (2, coefficients.shape[-1])
                    or not np.all((occupations == 0) | (occupations == 1))):
                raise ValueError("Unrestricted orbitals require two integer 0/1 occupation rows.")
            self.occupations = occupations
        else:
            if (occupations.shape != (coefficients.shape[-1],)
                    or not np.all((occupations == 0) | (occupations == 2))):
                raise ValueError("Restricted orbitals require integer 0/2 occupations.")
            self.occupations = np.stack([occupations / 2] * 2)
        seed_backends = ("full", "direct") if jk_backend == "direct" else ("full",)
        if not mf.mol.cart or getattr(mf, "jk_backend", "full") not in seed_backends:
            raise NotImplementedError("Use Cartesian AOs and full J/K, or explicitly opt into direct HF J/K.")
        self.jk_backend = jk_backend
        self.direct_scf_workers = getattr(mf, "direct_scf_workers", 1)
        self.xc = mf.xc
        self.rsh = rsh_preset(self.xc)
        self.kind = xc_type(local_xc_spec(self.xc))
        if self.kind not in ("HF", "LDA", "GGA"):
            raise NotImplementedError("This molecular adapter supports HF/LDA/GGA/global hybrids.")
        if jk_backend == "direct" and (self.kind != "HF" or self.rsh is not None):
            raise NotImplementedError("Direct molecular response currently supports HF only.")
        self.spec = mf.mol.to_spec()
        self.basis = mf.mol.basis
        self.coordinates = jnp.asarray(self.spec.coords_bohr)
        self.coefficients = jnp.asarray(coefficients)
        self.grids_level = mf.grids_level
        self.max_l = mf.max_l
        self.gradient_tolerance = gradient_tolerance
        self.differentiation = scf.SCFDifferentiationConfig(
            mode=mode, tolerance=1e-10, max_iter=100, restart=40)
        topology, self.parameters = integrals.prepare_basis(self.spec, self.basis, cart=True)
        self.plan = integrals.make_plan(topology, backend="native")
        distances = np.linalg.norm(np.asarray(self.parameters.centers)[:, None]
                                   - np.asarray(self.coordinates)[None], axis=-1)
        self.owners = np.argmin(distances, axis=1)
        if np.max(distances[np.arange(len(self.owners)), self.owners]) > 1e-12:
            raise ValueError("Every basis center must belong to a molecular atom.")

    def inputs(self, coordinates, field=None, *, potential=None):
        """Assemble differentiable inputs, optionally adding a real AO potential.

        The physical one-body perturbation is sym(potential), in Hartree.
        This defines a symmetric full-matrix coordinate convention for energy
        derivatives; an antisymmetric argument has zero physical effect.
        The supplied AO matrix is held fixed with respect to geometry unless
        the caller explicitly supplies its coordinate dependence.
        ``field=None`` omits electric-field assembly. An explicit zero array
        keeps the complete field response live for differentiation.
        """
        coordinates = jnp.asarray(coordinates)
        has_field = field is not None
        field = jnp.asarray(field) if has_field else None
        if coordinates.shape != self.coordinates.shape or (has_field and field.shape != (3,)):
            raise ValueError("Expected (natom,3) coordinates and a three-component field.")
        if jnp.iscomplexobj(coordinates) or (has_field and jnp.iscomplexobj(field)):
            raise ValueError("Coordinates and electric fields must be real.")
        if potential is not None:
            potential = jnp.asarray(potential)
            nao = self.coefficients.shape[-2]
            if potential.shape != (nao, nao) or jnp.iscomplexobj(potential):
                raise ValueError("AO potential must be a real (nao, nao) matrix.")
        moving = replace(self.parameters, centers=coordinates[self.owners],
                         nuclear_coords=coordinates)
        # IntegralPlan defaults to a moving nuclear-charge-center origin.
        # A field Hamiltonian needs the same fixed origin for electrons/nuclei.
        if has_field:
            dipole = self.plan.evaluate("dipole", moving, origin=jnp.zeros(3))
        extra = {}
        if self.kind == "HF":
            nao = self.coefficients.shape[-2]
            overlap = self.plan.evaluate("overlap", moving)
            hcore = self.plan.evaluate("kinetic", moving) + self.plan.evaluate("nuclear", moving)
            if self.jk_backend == "direct":
                from gradscf.integrals.backends.native.jk import NativeDirectBasis
                eri = None
                extra = dict(direct_basis=NativeDirectBasis(self.plan, moving))
            else:
                eri = self.plan.evaluate("eri", moving)
            enuc = scf.nuclear_repulsion_energy(coordinates, jnp.asarray(self.spec.charges))
            ao, dao, weights = jnp.zeros((0, nao)), jnp.zeros((4, 0, nao)), jnp.zeros(0)
        else:
            build = build_uks_integral_inputs if self.unrestricted else build_rks_integral_inputs
            options = {} if self.unrestricted else dict(include_dipole_integrals=False)
            data = build(
                atom=replace(self.spec, coords_bohr=coordinates), basis=self.basis,
                xc_spec=self.xc, grids_level=self.grids_level, max_l=self.max_l,
                integral_backend="native", cart=True, **options)
            overlap, hcore = data.overlap, data.hcore
            eri = data.eri if data.eri is not None else data.eri_pair_matrix
            enuc = data.nuclear_repulsion
            ao, dao, weights = data.ao, data.ao_deriv1, data.grid_weights
            if self.rsh is not None:
                extra = dict(eri_lr=data.eri_lr, dispersion_energy=data.dispersion_energy)
        if potential is not None:
            hcore = hcore+.5*(potential+potential.T)
        if has_field:
            hcore = hcore+jnp.einsum("x,xpq->pq", field, dipole)
            enuc = enuc-field@(jnp.asarray(self.spec.charges) @ coordinates)
        return dict(overlap=overlap, hcore=hcore, eri=eri, nuclear_repulsion=enuc,
                    ao=ao, ao_deriv1=dao, grid_weights=weights, **extra)

    def solve(self, coordinates, field=None, *, potential=None):
        """Return the relaxed array-valued state, retaining its implicit response."""
        return self.solve_with_inputs(coordinates, field, potential=potential)[0]

    def solve_with_inputs(self, coordinates, field=None, *, potential=None):
        """Return ``(state, inputs)`` from one assembly for downstream response."""
        inputs = self.inputs(coordinates, field, potential=potential)
        minimize = scf.minimize_uks_from_integrals if self.unrestricted else scf.minimize_roks_from_integrals
        options = dict(direct_scf_workers=self.direct_scf_workers) if self.jk_backend == "direct" else {}
        state = minimize(
            **inputs, xc_spec=self.xc,
            mo_coeff=self.coefficients, mo_occ=self.occupations,
            orthonormalize_initial=True, max_iterations=300,
            gradient_tolerance=self.gradient_tolerance, differentiation=self.differentiation, **options)
        return state, inputs

    def __call__(self, coordinates, field=None, *, potential=None):
        state = self.solve(coordinates, field, potential=potential)
        return jnp.where(state.stationary, state.total_energy, jnp.nan)

    def dipole(self, coordinates, field=None, *, potential=None):
        """Total dipole from the stationary first-field-derivative identity.

        The field enters this adapter only through h(F)=h(0)+F.r and the
        nuclear term -F.sum(Z*R). At stationarity, -dE/dF is therefore
        sum(Z*R)-Tr[D*r]. D remains the fully differentiable relaxed density:
        further field/geometry derivatives include its implicit response.
        This avoids building higher state derivatives multiplied by the
        vanishing stationarity residual in a direct nested energy graph.
        """
        coordinates = jnp.asarray(coordinates)
        state = self.solve(coordinates, field, potential=potential)
        moving = replace(self.parameters, centers=coordinates[self.owners],
                         nuclear_coords=coordinates)
        position = self.plan.evaluate("dipole", moving, origin=jnp.zeros(3))
        density = jnp.sum(state.density_matrix, axis=0)
        value = (jnp.asarray(self.spec.charges) @ coordinates
                 - jnp.einsum("pq,xqp->x", density, position))
        return jnp.where(state.stationary, value, jnp.nan)
