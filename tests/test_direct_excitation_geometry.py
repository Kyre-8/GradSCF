"""Relaxed molecular excitation gradients, distances in bohr and energies in Ha.

Each displaced point rebuilds the native integrals, atom-centered quadrature,
SCF density/orbitals and response operator. Full/direct use the same SCF
algorithm; the small full-ERI path is only an independent storage oracle.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, tdscf
from gradscf.data.molecule import parse_molecule_spec
from gradscf.scf import DifferentiableSCF, DifferentiableSCFConfig, SCFDifferentiationConfig
from gradscf.scf.core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer
from gradscf.scf.inputs import build_rks_integral_inputs, build_uks_integral_inputs
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
from gradscf.scf.rsh import rsh_preset
from gradscf.scf.uks import UKSConfig, run_uks_from_integrals
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
from gradscf.tddft._unrestricted_semilocal_response import UnrestrictedSemilocalResponseFunctional


def _global_network():
    layers = tuple((jnp.linspace(-.17, .21, ni*no).reshape(ni, no),
                    jnp.linspace(.01, .03, no))
                   for ni, no in ((4, 5), (5, 4), (4, 3), (3, 1)))

    def network(inputs):
        value = inputs
        for weight, bias in layers[:-1]:
            value = jnp.tanh(value @ weight + bias)
        weight, bias = layers[-1]
        return (value @ weight + bias)[..., 0]

    def energy(params, inputs):
        density, ao, weights = inputs
        rho = jnp.einsum("sgp,gp->sg", jnp.matmul(ao, density), ao)
        features = jnp.stack((rho[0], rho[1], rho[0]*rho[1], (rho[0]+rho[1])**2), axis=-1)
        pooled = jnp.dot(weights, (rho[0]+rho[1])*network(features))
        joint = jnp.stack((jnp.trace(density[0]), jnp.trace(density[1]),
                           jnp.sum(density[0]), jnp.sum(density[1])))
        return params["scale"]*(pooled + .07*pooled**2 + .13*network(joint))

    return dft.custom_xc(
        input_fn=lambda state: (state.density_matrix, state.ao, state.weights),
        energy_fn=energy, name="geometry_joint_spin_global"), {"scale": jnp.asarray(.003)}


def _assemble(spec, distance, unrestricted, xc_spec, direct, *, basis="sto-3g"):
    coordinates = jnp.asarray(spec.coords_bohr).at[1, 2].set(distance)
    options = dict(xc_spec=xc_spec, jk_backend="direct" if direct else "full",
                   direct_scf_tol=0., direct_scf_workers=1, max_cycle=100,
                   conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-8,
                   damping=.1)
    config = UKSConfig(**options) if unrestricted else RKSConfig(**options)
    build = build_uks_integral_inputs if unrestricted else build_rks_integral_inputs
    controls = {} if unrestricted else {"include_dipole_integrals": False}
    inputs = build(atom=replace(spec, coords_bohr=coordinates), basis=basis,
        config=config, xc_spec=xc_spec, grids_level=0, cart=True,
        integral_backend="native", libcint_geometry_grad_policy="analytic", **controls)
    if direct:
        assert inputs.direct_basis is not None
        assert inputs.eri is None or inputs.eri.size == 0
        assert inputs.eri_lr is None
        assert getattr(inputs, "eri_pair_matrix", None) is None
    return inputs, config, coordinates


def _reference(inputs, coordinates, unrestricted, *, coefficients, occupations,
               energies, density, fock, xc_spec, total_energy=0.):
    source = inputs.eri
    if source is None:
        source = inputs.eri_pair_matrix
    empty = jnp.zeros((0, 0, 0, 0), dtype=inputs.hcore.dtype)
    source = empty if source is None else source
    preset = rsh_preset(xc_spec)
    common = dict(
        ao=inputs.ao, ao_deriv1=inputs.ao_deriv1, ao_laplacian=inputs.ao_laplacian,
        grid=QuadratureGrid(inputs.grid_weights, inputs.coords),
        dipole_integrals=(jnp.zeros((3, inputs.hcore.shape[0], inputs.hcore.shape[0]))
                         if inputs.dipole_integrals is None else inputs.dipole_integrals),
        h1e=inputs.hcore, overlap_matrix=inputs.overlap,
        nuclear_repulsion=inputs.nuclear_repulsion, atom_coords=coordinates,
        direct_basis=inputs.direct_basis, mf_energy=total_energy,
        exact_exchange_fraction=1. if xc_spec == "hf" else 0., xc_spec=xc_spec,
        eri_lr=inputs.eri_lr, dispersion_energy=inputs.dispersion_energy,
        range_separated_coefficients=(None if preset is None else
                                      preset.to_range_separated_hybrid_coefficients()),
    )
    if unrestricted:
        return UnrestrictedMolecule(**common, rep_tensor=source,
            mo_coeff=coefficients, mo_occ=occupations, mo_energy=energies,
            rdm1=density, fock_matrix=fock,
            nocc_alpha=inputs.nalpha, nocc_beta=inputs.nbeta)
    return RestrictedMolecule(**common,
        rep_tensor=empty if source.ndim == 2 else source,
        eri_pair_matrix=source if source.ndim == 2 else None,
        mo_coeff=jnp.stack((coefficients, coefficients)),
        mo_occ=jnp.stack((.5*occupations, .5*occupations)),
        mo_energy=jnp.stack((energies, energies)),
        rdm1=jnp.stack((.5*density, .5*density)), fock_matrix=fock,
        nocc=inputs.nelectron//2)


def _classical_reference(inputs, config, coordinates, unrestricted, xc_spec):
    if unrestricted:
        solved = run_uks_from_integrals(**inputs.as_uks_kwargs(), config=config)
        state = _reference(inputs, coordinates, True,
            coefficients=jnp.stack((solved.mo_coeff_alpha, solved.mo_coeff_beta)),
            occupations=jnp.stack((solved.mo_occ_alpha, solved.mo_occ_beta)),
            energies=jnp.stack((solved.mo_energy_alpha, solved.mo_energy_beta)),
            density=jnp.stack((solved.density_matrix_alpha, solved.density_matrix_beta)),
            fock=jnp.stack((solved.fock_matrix_alpha, solved.fock_matrix_beta)),
            xc_spec=xc_spec, total_energy=solved.total_energy)
    else:
        solved = run_rks_from_integrals_traceable(**inputs.as_rks_kwargs(), config=config)
        state = _reference(inputs, coordinates, False,
            coefficients=solved.mo_coeff, occupations=solved.mo_occ,
            energies=solved.mo_energy, density=solved.density_matrix,
            fock=solved.fock_matrix, xc_spec=xc_spec, total_energy=solved.total_energy)
    return state, solved.converged


def _neural_reference(inputs, coordinates, unrestricted, model, params):
    energies, coefficients = _diagonalize_fock(inputs.hcore, _orthogonalizer(inputs.overlap, 1e-10))
    nao = coefficients.shape[0]
    if unrestricted:
        occupations = jnp.zeros((2, nao)).at[0, :inputs.nalpha].set(1.).at[1, :inputs.nbeta].set(1.)
        coefficients = jnp.stack((coefficients, coefficients))
        energies = jnp.stack((energies, energies))
        density = jax.vmap(_build_density_from_occ)(coefficients, occupations)
    else:
        occupations = jnp.zeros(nao).at[:inputs.nelectron//2].set(2.)
        density = _build_density_from_occ(coefficients, occupations)
    initial = _reference(inputs, coordinates, unrestricted, coefficients=coefficients,
        occupations=occupations, energies=energies, density=density, fock=None, xc_spec=None)
    solver = DifferentiableSCF(DifferentiableSCFConfig(
        mode="self_consistent", gradient_mode="implicit", max_cycle=100, damping=.15,
        conv_tol_energy=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-8,
        eigenvalue_jitter=0., vxc_clip=float("inf"),
        differentiation=SCFDifferentiationConfig(mode="implicit", tolerance=1e-10,
            max_iter=50, restart=20, require_converged=True)))
    state, info = solver.run(initial, model, params)
    if unrestricted:
        fock = solver._unrestricted_fock_from_density(
            molecule=state, xc_functional=model, params=params, density_spin=state.rdm1,
            mo_coeff_spin=state.mo_coeff, mo_occ_spin=state.mo_occ, mo_energy_spin=state.mo_energy,
            h1e=state.h1e, rep_tensor=state.rep_tensor, df_factors=None,
            weights=state.grid.weights, with_exchange=False, jk_from_orbitals=False)[0]
    else:
        context = solver._restricted_scf_context(state, model)
        fock = solver._restricted_fock_from_density(context, params, state.rdm1.sum(axis=0),
            state.mo_coeff[0], state.mo_energy[0], with_k=False)[0]
    return replace(state, fock_matrix=fock), info.converged


def _excitation(spec, distance, unrestricted, xc_spec, full_td, direct, model, params, *, basis="sto-3g"):
    inputs, config, coordinates = _assemble(spec, distance, unrestricted,
        "hf" if xc_spec == "network" else xc_spec, direct, basis=basis)
    if xc_spec == "network":
        state, ground_converged = _neural_reference(inputs, coordinates, unrestricted, model, params)
        functional, response_params = model, params
    else:
        state, ground_converged = _classical_reference(inputs, config, coordinates, unrestricted, xc_spec)
        functional = (UnrestrictedSemilocalResponseFunctional(xc_spec) if unrestricted
                      else SemilocalResponseFunctional(xc_spec))
        response_params = None
    driver = (tdscf.TDDFT if full_td else tdscf.TDA)(
        state, xc_functional=functional, xc_params=response_params, nstates=1,
        davidson_tol=1e-10, davidson_max_iter=100,
        response_kernel_options={"max_memory": 128., "integral_cache": "none",
            "direct_tol": 0., "direct_workers": 1, "grid_block_size": 128})
    excited = driver.kernel()
    converged = jnp.asarray(ground_converged) & jnp.asarray(excited.converged)
    energy = jnp.where(converged, excited.excitation_energies[0], jnp.nan)
    return energy, converged, state.rdm1, state.mo_coeff


def _check_relaxed_excitation_geometry(spec, unrestricted, xc_spec, full_td, *,
                                      basis="sto-3g", expected_ao=None):
    model, params = _global_network()
    full = jax.jit(lambda bond: _excitation(spec, bond, unrestricted, xc_spec,
                                           full_td, False, model, params, basis=basis))
    direct = jax.jit(lambda bond: _excitation(spec, bond, unrestricted, xc_spec,
                                             full_td, True, model, params, basis=basis))
    point, step = jnp.asarray(1.6), 2e-4
    reference, converged, density, coefficients = full(point)
    assert bool(converged) and np.isfinite(reference)
    if expected_ao is not None:
        assert density.shape == coefficients.shape == (2, expected_ao, expected_ao)
    full_value = lambda bond: full(bond)[0]
    reference_gradient = jax.grad(full_value)(point)
    reference_jvp = jax.jvp(full_value, (point,), (jnp.asarray(1.),))[1]
    plus, minus = full(point+step), full(point-step)
    assert bool(plus[1]) and bool(minus[1])
    reference_fd = (plus[0]-minus[0])/(2*step)
    assert jnp.linalg.norm(plus[2]-minus[2]) > 1e-8, "The SCF density must move with the geometry"
    assert jnp.linalg.norm(plus[3]-minus[3]) > 1e-8, "The orbitals must be solved at every geometry"
    np.testing.assert_allclose(reference_gradient, reference_jvp, atol=3e-8, rtol=2e-6)
    np.testing.assert_allclose(reference_gradient, reference_fd, atol=3e-6, rtol=3e-4)

    # Only after the native-full oracle is checked do we enter the direct AD
    # path, which initially exposes the assembly/native-geometry support gap.
    observed, converged, _, _ = direct(point)
    assert bool(converged)
    gradient = jax.grad(lambda bond: direct(bond)[0])(point)
    tangent = jax.jvp(lambda bond: direct(bond)[0], (point,), (jnp.asarray(1.),))[1]
    np.testing.assert_allclose(observed, reference, atol=3e-9, rtol=0)
    np.testing.assert_allclose(gradient, reference_gradient, atol=8e-7, rtol=3e-5)
    np.testing.assert_allclose(tangent, reference_jvp, atol=8e-7, rtol=3e-5)
    plus, minus = direct(point+step), direct(point-step)
    assert bool(plus[1]) and bool(minus[1])
    np.testing.assert_allclose(gradient, (plus[0]-minus[0])/(2*step), atol=3e-6, rtol=3e-4)
    return dict(excitation_ha=float(observed), derivative_ha_per_bohr=float(gradient),
        energy_error_ha=float(jnp.abs(observed-reference)),
        derivative_error_ha_per_bohr=float(jnp.abs(gradient-reference_gradient)),
        direct_fd_error_ha_per_bohr=float(jnp.abs(gradient-(plus[0]-minus[0])/(2*step))),
        jvp_vjp_error_ha_per_bohr=float(jnp.abs(gradient-tangent)))


@pytest.mark.parametrize("unrestricted,xc_spec,full_td", [
    (False, "hf", True), (False, "pbe", False),
    (True, "hf", True), (True, "pbe", False),
    (False, "network", True), (True, "network", True),
], ids=["rhf-tdhf", "rks-pbe-tda", "uhf-utdhf", "uks-pbe-utda", "global-nn-rtd", "global-nn-utd"])
def test_relaxed_direct_excitation_geometry_matches_full_and_finite_difference(
        unrestricted, xc_spec, full_td):
    # H2 / H2+ at 1.6 bohr, Cartesian STO-3G, atom-centered level-0 grid.
    spec = parse_molecule_spec("H 0 0 0; H 0 0 1.6", unit="Bohr",
                               charge=int(unrestricted), spin=int(unrestricted))
    _check_relaxed_excitation_geometry(spec, unrestricted, xc_spec, full_td)


@pytest.mark.parametrize("xc_spec,full_td", [("hf", True), ("pbe", True), ("network", True)],
                         ids=["hf-utdhf", "pbe-utddft", "global-nn-utddft"])
def test_heh_both_spin_channels_relaxed_geometry_matches_full_and_finite_difference(xc_spec, full_td):
    # Neutral HeH doublet: 4 Cartesian s AOs, alpha 2 occupied/2 virtual,
    # beta 1 occupied/3 virtual. Both spin response channels are nonempty.
    spec = parse_molecule_spec("He 0 0 0; H 0 0 1.6", unit="Bohr", charge=0, spin=1)
    assert spec.nelectron == 3 and spec.spin == 1
    _check_relaxed_excitation_geometry(spec, True, xc_spec, full_td, basis="6-31g", expected_ao=4)


def test_restricted_pbe_full_tddft_relaxed_geometry_matches_full_and_finite_difference():
    spec = parse_molecule_spec("H 0 0 0; H 0 0 1.6", unit="Bohr", charge=0, spin=0)
    _check_relaxed_excitation_geometry(spec, False, "pbe", True)


def _pyscf_excitation(spec, distance, xc_spec, unrestricted, basis):
    from pyscf import dft as pyscf_dft, gto

    inputs, _, coordinates = _assemble(spec, jnp.asarray(distance), unrestricted,
                                        xc_spec, True, basis=basis)
    molecule = gto.M(atom=list(zip(spec.symbols, np.asarray(coordinates))),
        basis=basis, unit="Bohr", charge=spec.charge, spin=spec.spin, cart=True, verbose=0)
    mf = (pyscf_dft.UKS if unrestricted else pyscf_dft.RKS)(molecule)
    mf.xc = {"svwn": "LDA_X,LDA_C_VWN", "lc-wpbe": "LC_WPBE"}.get(xc_spec, xc_spec)
    # Share the quadrature, not the AO/XC/SCF/TD implementation. At each
    # displaced geometry GradSCF rebuilds this moving atom-centered grid.
    mf.grids.coords = np.asarray(inputs.coords)
    mf.grids.weights = np.asarray(inputs.grid_weights)
    mf.small_rho_cutoff = 0.
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    response = mf.TDDFT()
    response.nstates = 1
    response.conv_tol = 1e-10
    energy, _ = response.kernel()
    assert np.all(response.converged)
    return float(energy[0])


@pytest.mark.parametrize("xc_spec", ["svwn", "pbe", "pbe0", "wb97xd"],
                         ids=["lda-svwn", "gga-pbe", "hybrid-pbe0", "rsh-wb97xd"])
@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
def test_functional_families_full_tddft_direct_geometry(xc_spec, unrestricted, record_property):
    # HeH exercises both spin channels for LDA/GGA/hybrid. The original
    # omegaB97X-D CHG tables support H/C/N/O/F, so use H2/H2+ for RSH.
    open_heh = unrestricted and xc_spec != "wb97xd"
    spec = parse_molecule_spec("He 0 0 0; H 0 0 1.6" if open_heh else
                               "H 0 0 0; H 0 0 1.6", unit="Bohr", spin=int(unrestricted),
                               charge=int(unrestricted and not open_heh))
    basis = "6-31g" if open_heh else "sto-3g"
    record_property("execution_mode", "pipeline_jit")
    metrics = _check_relaxed_excitation_geometry(spec, unrestricted, xc_spec, True,
        basis=basis, expected_ao=4 if open_heh else None)
    for name, value in metrics.items():
        record_property(name, value)
    pytest.importorskip("pyscf")
    reference = lambda distance: _pyscf_excitation(spec, distance, xc_spec, unrestricted, basis)
    energy = reference(1.6)
    derivative = (reference(1.6002)-reference(1.5998))/.0004
    record_property("pyscf_energy_error_ha", abs(metrics["excitation_ha"]-energy))
    record_property("pyscf_derivative_error_ha_per_bohr", abs(metrics["derivative_ha_per_bohr"]-derivative))
    np.testing.assert_allclose(metrics["excitation_ha"], energy, atol=3e-7, rtol=0)
    np.testing.assert_allclose(metrics["derivative_ha_per_bohr"], derivative, atol=4e-6, rtol=4e-4)


def test_uks_static_occupation_validation_retains_jit_and_eager_errors():
    from gradscf.scf.uks import _validate_spin_occ

    validate = lambda occupations: _validate_spin_occ(
        occupations, nao=2, nelec=1, label="alpha")
    with pytest.raises(ValueError, match="lie in"):
        validate(jnp.asarray([1.1, -.1]))
    with pytest.raises(ValueError, match="sum to"):
        validate(jnp.asarray([.4, .2]))
    with pytest.raises(ValueError, match="length nao"):
        validate(jnp.asarray([1., 0., 0.]))
    occupations = jnp.asarray([1., 0.])
    np.testing.assert_array_equal(jax.jit(validate)(occupations), occupations)
