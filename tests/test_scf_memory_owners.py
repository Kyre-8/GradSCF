"""Retained initial energy and raw molecular metadata remain caller-owned."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf._memory import tree_storage_bytes
from gradscf.scf import orbital_optimization as orbital
from gradscf_tools.molecular_response import MolecularEnergy
from test_orbital_memory_budget import _inputs


def test_molecular_declares_raw_charge_and_coordinate_owners():
    energy = MolecularEnergy.__new__(MolecularEnergy)
    energy.coefficients = jax.ShapeDtypeStruct((2, 2), jnp.float64)
    energy.occupations = np.zeros((2, 2))
    energy.parameters = ()
    energy.coordinates = jax.ShapeDtypeStruct((1, 3), jnp.float64)
    energy.owners = np.zeros(1, dtype=int)
    energy.plan = SimpleNamespace(topology=None)
    energy.spec = SimpleNamespace(charges=jax.ShapeDtypeStruct((257,), jnp.float64),
        coords_bohr=jax.ShapeDtypeStruct((257, 3), jnp.float64))
    declared = tree_storage_bytes(energy._memory_inputs(), exclude=(energy.coefficients,
        energy.occupations, energy.parameters, energy.coordinates, energy.owners))
    assert declared == 257*4*8


def test_response_reservation_keeps_initial_energy(monkeypatch):
    energy = jnp.asarray(2.75)
    declarations = []
    original = orbital.tree_storage_bytes
    def count(*trees, **kwargs):
        declarations.append(trees)
        return original(*trees, **kwargs)
    monkeypatch.setattr(orbital, 'tree_storage_bytes', count)
    def problem(*args, **kwargs):
        return (1, lambda angles, base: base,
            lambda base: jnp.zeros((2, 2, 2)),
            lambda density, inputs: (energy, jnp.zeros((2, 2, 2))),
            lambda angles, base, inputs: energy+jnp.sum(angles**2))
    monkeypatch.setattr(orbital, '_problem_functions', problem)
    def forward(args, coeff, **kwargs):
        return SimpleNamespace(coeff=coeff, gradient_norm=jnp.asarray(0.),
            iterations=jnp.asarray(0), evaluations=jnp.asarray(1), stages=(), polishing=())
    monkeypatch.setattr(orbital, 'solve_orbitals', forward)
    class Captured(Exception):
        pass
    def checked(*args, **kwargs):
        assert any(value is energy for call in declarations for value in call), (
            'The retained initial_energy owner is missing from response reservation')
        raise Captured
    monkeypatch.setattr(orbital, 'attach_scf_backward', checked)
    with pytest.raises(Captured):
        orbital.minimize_uks_from_integrals(**_inputs(), max_memory=8., reserved_bytes=4096)
