"""Original omegaB97X-D: full/LR exchange and separate CHG bookkeeping."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import Avogadro, physical_constants

from gradscf.scf.rks import RKSConfig, run_rks_from_integrals
from gradscf.scf.uks import UKSConfig, run_uks_from_integrals


def chg_oracle(mol):
    table = {"H": (.14, 1.001), "O": (.70, 1.342)}
    distance_nm = np.linalg.norm(mol.atom_coords()[:, None] - mol.atom_coords()[None], axis=-1)
    distance_nm *= physical_constants["Bohr radius"][0] / 1e-9
    energy = 0.
    for i in range(mol.natm):
        for j in range(i):
            ci, ri = table[mol.atom_symbol(i)]
            cj, rj = table[mol.atom_symbol(j)]
            r = distance_nm[i, j]
            energy -= np.sqrt(ci*cj) / r**6 / (1 + 6*(r/((ri+rj)*.1))**-12)
    return energy / Avogadro / physical_constants["Hartree energy"][0]


@pytest.fixture(scope="module", params=[False, True], ids=["water-rks", "oh-uks"])
def reference(request):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft
    unrestricted = request.param
    atom = "O 0 0 0; H .1 .0 .97" if unrestricted else "O 0 0 0; H 0 .76 .58; H 0 -.76 .58"
    mol = pyscf.gto.M(atom=atom, basis="sto-3g", cart=True, spin=int(unrestricted), verbose=0)
    mf = dft.UKS(mol) if unrestricted else dft.RKS(mol)
    # This spelling supplies the electronic LibXC 471 functional only.
    mf.xc = "wb97xd"
    mf.grids.level = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    dao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
    with mol.with_range_coulomb(.2):
        lr = mol.intor("int2e")
    kwargs = dict(overlap=jnp.asarray(mf.get_ovlp()), hcore=jnp.asarray(mf.get_hcore()),
                  eri=jnp.asarray(mol.intor("int2e")), eri_lr=jnp.asarray(lr),
                  dispersion_energy=jnp.asarray(chg_oracle(mol)), nuclear_repulsion=mol.energy_nuc(),
                  ao=jnp.asarray(dao[0]), ao_deriv1=jnp.asarray(dao),
                  grid_weights=jnp.asarray(mf.grids.weights))
    config_type = UKSConfig if unrestricted else RKSConfig
    kwargs["config"] = config_type(xc_spec="wb97xd", max_cycle=80, conv_tol=1e-11,
                                   conv_tol_density=1e-9, conv_tol_grad=1e-8)
    if unrestricted:
        kwargs.update(nalpha=mol.nelec[0], nbeta=mol.nelec[1],
                      init_density_alpha=mf.make_rdm1()[0], init_density_beta=mf.make_rdm1()[1])
    else:
        kwargs.update(nelectron=mol.nelectron, init_density=mf.make_rdm1())
    return unrestricted, mol, mf, kwargs


def test_full_method_matches_pyscf_electronic_plus_independent_chg(reference):
    unrestricted, mol, mf, kwargs = reference
    run = run_uks_from_integrals if unrestricted else run_rks_from_integrals
    result = run(**kwargs)
    assert bool(result.converged)
    np.testing.assert_allclose(result.total_energy, mf.e_tot + chg_oracle(mol), atol=2e-8, rtol=0)
    np.testing.assert_allclose(result.electronic_energy, mf.e_tot - mol.energy_nuc(), atol=2e-8, rtol=0)
    np.testing.assert_allclose(result.nuclear_repulsion, mol.energy_nuc(), atol=1e-13)
    np.testing.assert_allclose(result.dispersion_energy, chg_oracle(mol), atol=1e-14)
    np.testing.assert_allclose(result.range_separated_coefficients, (.2, 1., .222036), atol=1e-14)
    np.testing.assert_allclose(result.exact_exchange_fraction, .222036, atol=1e-14)
    density = (jnp.stack([result.density_matrix_alpha, result.density_matrix_beta])
               if unrestricted else result.density_matrix)
    fock = (jnp.stack([result.fock_matrix_alpha, result.fock_matrix_beta])
            if unrestricted else result.fock_matrix)
    np.testing.assert_allclose(density, mf.make_rdm1(), atol=3e-7, rtol=0)
    np.testing.assert_allclose(fock, mf.get_fock(dm=mf.make_rdm1()), atol=2e-7, rtol=0)


@pytest.mark.parametrize("missing", ["eri_lr", "dispersion_energy"])
def test_named_method_rejects_missing_components(reference, missing):
    unrestricted, _, _, kwargs = reference
    run = run_uks_from_integrals if unrestricted else run_rks_from_integrals
    incomplete = dict(kwargs)
    incomplete.pop(missing)
    with pytest.raises(ValueError, match=missing):
        run(**incomplete)


def test_named_method_rejects_df(reference):
    unrestricted, _, _, kwargs = reference
    run = run_uks_from_integrals if unrestricted else run_rks_from_integrals
    with pytest.raises(NotImplementedError, match="(?i)(range|wb97|RSH).*full"):
        run(**dict(kwargs, config=replace(kwargs["config"], jk_backend="df")))


def test_named_native_input_assembly_and_response_boundary():
    from gradscf import dft, gto
    from gradscf.scf.inputs.assembly import build_rks_integral_inputs
    data = build_rks_integral_inputs(atom="O 0 0 0; H 0 .76 .58; H 0 -.76 .58",
                                     basis="sto-3g", xc_spec="wb97xd", grids_level=0)
    assert data.eri.shape == data.eri_lr.shape == (7, 7, 7, 7)
    assert data.eri_pair_matrix is None
    assert data.dispersion_energy < 0
    mf = dft.RKS(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"))
    mf.xc = "wb97xd"
    with pytest.raises(NotImplementedError, match="(?i)range.separated.*response"):
        mf.to_reference()
    from gradscf.scf.reference import reference_state_signature
    with pytest.raises(NotImplementedError, match="(?i)range.separated.*response"):
        reference_state_signature(mf)


@pytest.mark.parametrize("mode", ["implicit", "unrolled"])
def test_orbital_rsh_energy_and_density_keep_lr_and_chg_as_dynamic_inputs(reference, mode):
    from gradscf.scf.orbital_optimization import minimize_roks_from_integrals, minimize_uks_from_integrals
    unrestricted, _, mf, inputs = reference
    run = minimize_uks_from_integrals if unrestricted else minimize_roks_from_integrals
    n = inputs["overlap"].shape[0]
    rotation = np.eye(n)
    angle = .17
    rotation[0, 0] = rotation[-1, -1] = np.cos(angle)
    rotation[0, -1], rotation[-1, 0] = -np.sin(angle), np.sin(angle)
    seed = np.asarray(mf.mo_coeff) @ rotation
    occ = np.asarray(mf.mo_occ) if unrestricted else np.stack([mf.mo_occ / 2]*2)
    fixed = {key: inputs[key] for key in ("overlap", "hcore", "eri", "nuclear_repulsion",
                                         "ao", "ao_deriv1", "grid_weights")}
    probe = jnp.asarray(np.random.default_rng(941).normal(size=(n, n)))

    def solve(p):
        return run(**fixed, eri_lr=(1+p[0])*inputs["eri_lr"],
                   dispersion_energy=inputs["dispersion_energy"]+p[1],
                   xc_spec="wb97xd", mo_coeff=seed, mo_occ=occ,
                   max_iterations=100, gradient_tolerance=2e-9, gradient_mode=mode)

    def observables(p):
        result = solve(p)
        return jnp.stack([result.total_energy, jnp.sum(probe*result.density_matrix.sum(axis=0))])

    point = jnp.array([.03, .002])
    compiled = jax.jit(solve)
    state = compiled(point)
    assert bool(state.stationary), float(state.gradient_norm)
    print("RSH orbital forward", "UKS" if unrestricted else "RKS", mode,
          "gradient norm", float(state.gradient_norm), flush=True)
    actual = jax.jit(jax.jacrev(observables))(point)
    print("RSH orbital Jacobian", np.asarray(actual).tolist(), flush=True)
    step = 2e-4
    finite = []
    for direction in jnp.eye(2):
        plus, minus = compiled(point+step*direction), compiled(point-step*direction)
        assert bool(plus.stationary) and bool(minus.stationary)
        values = lambda result: jnp.stack([result.total_energy,
            jnp.sum(probe*result.density_matrix.sum(axis=0))])
        finite.append((values(plus)-values(minus))/(2*step))
    finite = jnp.stack(finite, axis=1)
    assert abs(float(finite[1, 0])) > 1e-7
    np.testing.assert_allclose(actual, finite, atol=3e-6, rtol=4e-5)
    np.testing.assert_allclose(actual[:, 1], [1., 0.], atol=1e-10, rtol=0)
    print("RSH orbital Jacobian maxabs AD-FD", float(jnp.max(jnp.abs(actual-finite))), flush=True)


def test_named_native_facades_match_independent_complete_energy(reference):
    from gradscf import dft, gto
    unrestricted, mol, mf_reference, _ = reference
    molecule = gto.M(atom=mol.atom, basis="sto-3g", spin=mol.spin, cart=True)
    cls = dft.UKS if unrestricted else dft.RKS
    # Keep the same stationary branch, including OH's nearly degenerate pi
    # orientations, while changing integral/grid/SCF implementations.
    mf = cls(molecule, xc="wb97xd", conv_tol=1e-11, conv_tol_grad=1e-8,
             init_guess=mf_reference.make_rdm1()).run()
    assert mf.converged
    expected = mf_reference.e_tot + chg_oracle(mol)
    np.testing.assert_allclose(mf.e_tot, expected, atol=2e-8, rtol=0)
    state = mf.reference if unrestricted else mf.scf_result
    np.testing.assert_allclose(state.range_separated_coefficients, (.2, 1., .222036), atol=1e-14)
    with pytest.raises(NotImplementedError, match="(?i)range.separated.*response"):
        mf.to_reference()
    print("native facade", "UKS" if unrestricted else "RKS", "E error Ha", float(mf.e_tot-expected))
