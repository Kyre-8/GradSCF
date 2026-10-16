"""Berry polarization response to atomic displacements and static electric fields.

Both adapters reuse the KS projector residual and shared solvers. The electric
field couples to the discrete Berry polarization through a scalar electric
enthalpy, not a periodic local AO potential or an XC-functional substitution.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np

from .core import Problem
from .periodic import periodic_problem
from .ks import ks_problem
from ..integrals.periodic.fft import build_inputs
from ..pbc.polarization import make_plan, ao_links, occupied_frame, polarization, _berry, _projected_frame
from ..pbc.scf import validate_reference
from ..pbc.kpoints import weights_for
from ..solvers.nonlinear.iterate import solve_nonlinear


def born_problem(source, *, tolerance=1e-8, gap_tol=1e-8, link_tol=1e-8):
    """Return a fixed-cell geometry Problem observing total P in e/bohr^2.

    response(tangent=dR).values['polarization'] is the relaxed directional
    dP/dR. Multiplication by cell volume gives the Born-charge contraction in
    units of e. The coordinates are Cartesian Bohr, the lattice/FFT topology,
    basis exponents and full equal-weight k mesh are fixed. Every explicit spin
    channel has integer unit occupations and an insulating indirect gap.

    The existing KS projector residual supplies density response. A local
    Cholesky frame and moving cross-k AO links supply the Berry derivative;
    canonical orbital eigenvectors are never differentiated. Reference phases
    anchor the local branch even when the principal forward phase is at pi.
    Build a fresh problem after changing the reference. Only first geometry
    response is certified by this adapter; Taylor/mixed field response is absent.
    """
    if getattr(source, 'temperature', None) is not None:
        raise NotImplementedError('Born response currently requires zero temperature.')
    validate_reference(source, integer=True, equal_weights=True, external=True)
    cell = source.cell
    counts = tuple(int(n) for n in cell.nelec)
    fractional = np.asarray(source.kpts)@np.asarray(cell.lattice).T/(2*np.pi)
    plan = make_plan(fractional, kweights=source.kweights)
    reference = polarization(source.inputs, source.result, plan, nocc=counts,
                             tolerance=tolerance, gap_tol=gap_tol, link_tol=link_tol)
    anchor = jax.tree.map(jax.lax.stop_gradient, reference.phases)
    base = periodic_problem(source.inputs, source.result, mesh=cell.mesh, nelec=counts,
        xc=source.xc, exxdiv=source.exxdiv, functional=source.functional,
        params=source.params, tolerance=tolerance)
    reference_frames = tuple(source.result.mo_coeff_spin[spin, :, :, :n] for spin, n in enumerate(counts))

    def args_at(coordinates):
        data = build_inputs(cell, parameters=cell.parameters_at(coordinates),
                            kpts=source.kpts, kweights=source.kweights)
        return (data._replace(hcore=data.hcore.astype(jnp.complex128)),)+base.inputs[1:]

    def quantities(state, coordinates):
        args = args_at(coordinates)
        data = args[0]
        values = base.quantities(state, args)
        density = values['density']
        frames = tuple(occupied_frame(density[spin], data.overlap, ref, tolerance=tolerance).coefficients
                       for spin, ref in enumerate(reference_frames))
        links = ao_links(data.ao, data.weights, data.coords, data.lattice, plan)
        berry = _berry(frames, links, data.lattice, plan, reference_phases=anchor, link_tol=link_tol)
        ionic = data.ionic_charges@data.atom_coords/jnp.linalg.det(data.lattice)
        return dict(values, polarization=berry.electronic+ionic)

    return Problem(lambda state,r:base.residual(state, args_at(r)), base.solution,
                   cell.coords, quantities, base.converged & reference.valid,
                   tolerance, series_supported=False)


def _density_polarization(density, data, reference_frames, plan, anchor):
    # The off-shell Hermitian extension is essential: enforcing QSQ=Q here
    # would invalidate the density gradient used to build the field Fock term.
    hermitian = .5*(density+density.conj().swapaxes(-1, -2))
    frames = tuple(_projected_frame(hermitian[spin], data.overlap, ref)[0]
                   for spin, ref in enumerate(reference_frames))
    links = ao_links(data.ao, data.weights, data.coords, data.lattice, plan)
    berry = _berry(frames, links, data.lattice, plan, reference_phases=anchor)
    return berry.electronic+data.ionic_charges@data.atom_coords/jnp.linalg.det(data.lattice)


def _field_term(density, data, field, reference_frames, plan, anchor):
    """Electric enthalpy contribution and its physical Hermitian AO Fock term.

    JAX's complex gradient g obeys dB=Re sum(g*dD), while our AO convention is
    dB=Re sum_k w_k Tr(F_B*dD). Thus F_B=Herm(g.T/w_k), with no extra spin factor.
    Each explicit spin density has unit occupation. No holomorphic assumption
    is appropriate for this real-valued, conjugation-dependent functional.
    """
    def energy(d):
        p = _density_polarization(d, data, reference_frames, plan, anchor)
        return -jnp.linalg.det(data.lattice)*jnp.dot(field, p)
    value, gradient = jax.value_and_grad(energy)(density)
    potential = gradient.swapaxes(-1, -2)/weights_for(data)[None, :, None, None]
    potential = .5*(potential+potential.conj().swapaxes(-1, -2))
    return value, potential


def _electric_model(data, state, *, nocc, evaluate, params=None, tolerance=1e-8,
                    reference_phases=None):
    """Explicit (data, model parameters, field) problem and Jacobi preconditioner.

    evaluate(density, data, params) returns the zero-field energy and Fock.
    This assembly also supports small independent finite-dimensional tests;
    it introduces no solver or alternative periodic electrostatics. An optional
    reference_phases fixes a supplied local branch across orbital-frame gauges;
    its shape and branch are validated by the ordinary polarization observer.
    """
    fractional = np.asarray(data.kpoints)@np.asarray(data.lattice).T/(2*np.pi)
    plan = make_plan(fractional, kweights=np.asarray(weights_for(data)))
    checked = polarization(data, state, plan, nocc=nocc, tolerance=tolerance,
                           reference_phases=reference_phases)
    anchor = jax.tree.map(jax.lax.stop_gradient,
                         checked.phases if reference_phases is None else reference_phases)
    coefficients = jnp.asarray(state.mo_coeff_spin, dtype=jnp.complex128)
    shape = state.density_spin.shape
    spins, nk, n, _ = shape
    frames = tuple(coefficients[spin, :, :, :no] for spin, no in enumerate(nocc))
    args = (data, params, jnp.zeros(3, dtype=jnp.float64))

    def energy(density, a):
        inputs, parameters, field = a
        zero_field = evaluate(density, inputs, parameters)[0]
        p = _density_polarization(density, inputs, frames, plan, anchor)
        return zero_field-jnp.linalg.det(inputs.lattice)*jnp.dot(field, p)

    def fock(density, a):
        inputs, parameters, field = a
        original = evaluate(density, inputs, parameters)[1]
        return original+_field_term(density, inputs, field, frames, plan, anchor)[1]

    base = ks_problem(coefficients.reshape(spins*nk, n, n),
        tuple(no for no in nocc for _ in range(nk)), args,
        lambda d,a:fock(d.reshape(shape), a).reshape(spins*nk, n, n),
        lambda d,a:energy(d.reshape(shape), a),
        overlap=lambda a:jnp.tile(a[0].overlap, (spins, 1, 1)),
        converged=state.converged & checked.valid, tolerance=tolerance)
    original_quantities = base.quantities

    def quantities(x, a):
        values = original_quantities(x, a)
        density = values['density'].reshape(shape)
        p = _density_polarization(density, a[0], frames, plan, anchor)
        number = jnp.sum(values['electron_number'].reshape(spins, nk)*weights_for(a[0]))
        return dict(values, density=density, polarization=p, electron_number=number,
                    electric_enthalpy=values['energy'])

    # This is only a numerical forward preconditioner. The physical residual
    # and its implicit derivative retain the complete coupled Jacobian.
    f0 = coefficients.conj().swapaxes(-1, -2)@evaluate(state.density_spin, data, params)[1]@coefficients
    diagonal_energy = jax.lax.stop_gradient(jnp.diagonal(f0, axis1=-2, axis2=-1).real)
    i, j = np.triu_indices(n, 1)
    blocks = []
    for spin, no in enumerate(nocc):
        occupied = jnp.arange(n) < no
        normal = jnp.broadcast_to(jnp.where(occupied, 1., -1.), (nk, n))
        spectrum = diagonal_energy[spin]
        gap = spectrum[:, j]-spectrum[:, i]
        upper = jnp.where(occupied[i] & occupied[j], 1.,
                          jnp.where((~occupied[i]) & (~occupied[j]), -1., gap))
        blocks.append(jnp.concatenate([normal, upper, upper], axis=1))
    diagonal = jax.lax.stop_gradient(jnp.concatenate(blocks).reshape(-1))
    return replace(base, quantities=quantities, series_supported=False), diagonal


def _field_problem(model):
    """Expose only the three Cartesian field components at a fixed reference."""
    def arguments(field):
        field = jnp.asarray(field)
        if field.shape != (3,) or jnp.iscomplexobj(field):
            raise ValueError('Use a real Cartesian field vector with shape (3,).')
        return model.inputs[:2]+(field,)
    return Problem(lambda x,e:model.residual(x, arguments(e)), model.solution,
        model.inputs[2], lambda x,e:model.quantities(x, arguments(e)),
        model.converged, model.tolerance, series_supported=False)


def _solve_electric(model, diagonal, args, *, config=None, linear_config=None):
    """Solve new (data, parameters, field) inputs in the fixed reference chart.

    The seed is certified at model.inputs, not at the new target inputs. A
    displaced geometry need not already have the old reference as its root.
    The shared solver checks the actual target residual and attaches its full
    implicit derivative; geometry/field response never freezes that state.
    """
    if not isinstance(args, tuple) or len(args) != 3:
        raise ValueError('Electric inputs must be (data, parameters, field).')
    field = jnp.asarray(args[2])
    if field.shape != (3,) or jnp.iscomplexobj(field):
        raise ValueError('Use a real Cartesian field vector with shape (3,).')
    args = args[:2]+(field,)
    initial, valid = model._base()
    valid &= jnp.all(jnp.isfinite(field)) & jnp.all(jnp.isfinite(diagonal) & (jnp.abs(diagonal)>1e-12))
    result = solve_nonlinear(lambda x:model.residual(x, args), initial,
        update=lambda x,r:x-r/diagonal, observable=lambda x:model.quantities(x, args)['energy'],
        config=config, linear_config=linear_config, valid_inputs=valid)
    values = model.quantities(result.solution, args)
    values = jax.tree.map(lambda value:value*jnp.where(result.converged, 1., jnp.nan), values)
    return result, values


def _solve_field(model, diagonal, field, *, config=None, linear_config=None):
    """Bounded fixed-geometry field continuation through the shared root."""
    return _solve_electric(model, diagonal, model.inputs[:2]+(field,),
                           config=config, linear_config=linear_config)


def electric_problem(source, *, tolerance=1e-8):
    """Zero-field static electric response with Cartesian field-vector inputs.

    response(tangent=e_beta).values['polarization'] is column beta of chi=dP/dE
    at fixed nuclei/cell, in atomic units. The dimensionless electronic dielectric
    tensor is epsilon_inf=I+4*pi*chi; the per-cell polarizability is volume*chi.
    Fields are in Eh/(e bohr). Integer occupations, full equal k weights and an
    insulating reference are required. No trajectory or canonical-eigenvector
    differentiation is used. Build a fresh problem after changing the reference.
    """
    if getattr(source, 'temperature', None) is not None:
        raise NotImplementedError('Electric response currently requires zero temperature.')
    validate_reference(source, integer=True, equal_weights=True, external=True)
    from ..pbc._kpoint import energy_and_fock
    evaluate = lambda d,a,p:energy_and_fock(d, a, mesh=source.cell.mesh, xc=source.xc,
        exxdiv=source.exxdiv, functional=source.functional, params=p)
    model, _ = _electric_model(source.inputs, source.result, nocc=source.cell.nelec,
        evaluate=evaluate, params=source.params, tolerance=tolerance)
    return _field_problem(model)


__all__ = ['born_problem', 'electric_problem']
