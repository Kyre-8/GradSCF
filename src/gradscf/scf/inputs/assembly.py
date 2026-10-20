"""Standalone integral/grid assembly for restricted and unrestricted SCF."""
from __future__ import annotations

from dataclasses import replace
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.integrals.basis import basis_from_molecule_spec, prepare_basis
from gradscf.integrals.plan import make_plan
from gradscf.integrals.grids import build_molecular_grid_from_spec
from gradscf.integrals.grids.ao import evaluate_cartesian_ao, evaluate_cartesian_ao_with_derivatives
from gradscf.integrals.backends.jax_reference import overlap_matrix, build_hcore, dipole_matrix, eri_tensor, eri_pair_matrix_packed, precompile_eri_kernels
from gradscf.scf.inputs.types import RKSIntegralInputs, UKSIntegralInputs
from gradscf.scf.core import _unrestricted_spin_electron_counts
from gradscf.scf.inputs import grid as _grid
from gradscf.data.molecule import MoleculeSpec, parse_molecule_spec
from gradscf.dft.xc import parse_recipe, parse_xc, xc_type
from gradscf.scf.rsh import local_xc_spec, rsh_preset


def _resolve_config(config, xc_spec, config_type):
    name = str(xc_spec if xc_spec is not None else (config.xc_spec if config is not None else "pbe"))
    parse_xc(local_xc_spec(name))
    cfg = config_type(xc_spec=name) if config is None else config
    return (replace(cfg, xc_spec=name) if cfg.xc_spec != name else cfg), name


def _prepare_basis_grid_context(**kwargs):
    return _grid._prepare_basis_grid_context(
        **kwargs, basis_from_molecule_spec=basis_from_molecule_spec,
        build_molecular_grid_from_spec=build_molecular_grid_from_spec,
        evaluate_cartesian_ao_with_derivatives=evaluate_cartesian_ao_with_derivatives)


def _grid_ao_payload(context, **kwargs):
    return _grid._grid_ao_payload(context, **kwargs,
        evaluate_cartesian_ao_with_derivatives=evaluate_cartesian_ao_with_derivatives)


def _build_common(*, atom, basis, cfg, xc_spec, unit, charge, spin, cart,
                  grids_level, max_l, grid_ao_backend, integral_backend,
                  geometry_grad_policy, include_dipole, precompile_eri,
                  precompile_eri_chunk_size, precompiler, chkfile, sap_basis,
                  chkfile_project, mol_kwargs):
    if not cart:
        raise NotImplementedError("SCF/grid assembly currently requires Cartesian AOs; integral plans also support spherical AOs")
    mode = str(integral_backend).lower()
    if mode in {"cpu", "libcint"}:
        mode = "native"
    if mode not in {"native", "jax"}:
        raise ValueError("integral_backend must be 'native'/'cpu' or 'jax'; the external GPU backend was removed")
    rsh = rsh_preset(xc_spec)
    recipe = parse_recipe(xc_spec)
    original_chg = recipe.dispersion is not None and rsh is not None and rsh.name == "wb97x-d"
    recipe.require_complete_support(provided=("dispersion",) if original_chg else ())
    if recipe.needs_laplacian:
        raise NotImplementedError("This molecular XC assembly requires an implemented density-Laplacian operator.")
    if rsh is not None and (mode != "native" or cfg.jk_backend not in {"full", "direct"}):
        raise NotImplementedError("Range-separated XC requires native full ERIs or direct J/K.")
    if grid_ao_backend != "jax":
        raise ValueError("Only grid_ao_backend='jax' is supported")
    if geometry_grad_policy not in {"analytic", "error", "zero"}:
        raise ValueError("geometry gradient policy must be 'analytic', 'error', or 'zero'")
    if chkfile is not None or sap_basis is not None or chkfile_project is not None:
        raise NotImplementedError("External checkpoint/SAP guess options were removed; supply an initial density matrix")
    if mol_kwargs:
        raise TypeError(f"Unsupported molecular options: {', '.join(sorted(mol_kwargs))}")
    spec = atom if isinstance(atom, MoleculeSpec) else parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
    laplacian = xc_type(local_xc_spec(xc_spec)) == "MGGA"
    needs_grid = not (xc_type(local_xc_spec(xc_spec)) == "HF" and np.any(np.asarray(spec.charges) == 0))
    context = _prepare_basis_grid_context(spec=spec, basis=basis, max_l=max_l,
        grids_level=grids_level, precompute_eri_groups=(mode == "jax"),
        needs_ao_laplacian=laplacian, needs_grid=needs_grid)
    if context.geometry_is_traced and geometry_grad_policy == "error":
        raise NotImplementedError("Geometry gradients are disabled by geometry_grad_policy='error'")
    eri_lr = dispersion_energy = None
    if original_chg:
        from gradscf.dft.dispersion import wb97xd_dispersion_energy
        dispersion_energy = wb97xd_dispersion_energy(spec.symbols, spec.coords_bohr)
    if mode == "native":
        top, params = prepare_basis(spec, basis, cart=True)
        if geometry_grad_policy == "zero":
            params = replace(params, centers=jax.lax.stop_gradient(params.centers),
                             nuclear_coords=jax.lax.stop_gradient(params.nuclear_coords))
        plan = make_plan(top, backend="native")
        s = plan.evaluate("overlap", params)
        h = plan.evaluate("kinetic", params)+plan.evaluate("nuclear", params)
        full_eri=pair=factors=None
        auxiliary=getattr(cfg,'auxbasis',None)
        if cfg.jk_backend=='direct':
            if context.geometry_is_traced and cfg.direct_scf_tol != 0.:
                raise ValueError('Direct-SCF geometry AD requires direct_scf_tol=0.')
        elif rsh is not None:
            full_eri = plan.evaluate("eri", params)
            eri_lr = plan.evaluate("eri", params, omega=rsh.default_omega)
        elif cfg.jk_backend=='df' and auxiliary is not None:
            if context.geometry_is_traced:raise NotImplementedError('Native auxiliary-integral geometry AD is not implemented.')
            from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan, unpack_factors
            aux_top,aux_params=prepare_basis(spec,auxiliary,cart=True)
            packed_factors=make_auxiliary_plan(top,aux_top).factors(params,aux_params,lindep=cfg.df_tol)
            factors=unpack_factors(packed_factors,top.nao)
        elif context.geometry_is_traced:
            # Preserve the existing native coordinate AD rule. The compact
            # output API is currently forward-only for integral parameters.
            full_eri=plan.evaluate('eri',params)
            rows,cols=np.tril_indices(top.nao)
            pair=full_eri[rows[:,None],cols[:,None],rows[None,:],cols[None,:]]
        else:
            pair=plan.evaluate('eri',params,aosym='s4')
        dipole = plan.evaluate("dipole", params) if include_dipole else None
    else:
        b = context.basis
        if precompile_eri:
            precompiler(b, engine="jit", chunk_size=int(precompile_eri_chunk_size))
        s, h = overlap_matrix(b), build_hcore(b)
        dipole = dipole_matrix(b) if include_dipole else None
        full_eri = None
        pair = None if cfg.jk_backend == "direct" else eri_pair_matrix_packed(b)
        factors=None
    if cfg.jk_backend == "df" and factors is None:
        from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors, eri_pair_matrix_to_df_factors_traceable
        factorize = eri_pair_matrix_to_df_factors_traceable if context.geometry_is_traced else eri_pair_matrix_to_df_factors
        factors = factorize(pair, nao=s.shape[0], tol=cfg.df_tol, max_rank=cfg.df_max_rank)
    ao, ao_deriv1, ao_laplacian = _grid_ao_payload(context, needs_ao_laplacian=laplacian)
    return spec, context, s, h, full_eri, pair, factors, dipole, ao, ao_deriv1, ao_laplacian, mode, eri_lr, dispersion_energy


def build_rks_integral_inputs(*, atom, basis, config=None, xc_spec=None,
    unit="Angstrom", charge=0, spin=0, cart=True, grids_level=0, max_l=3,
    grid_ao_backend="jax", integral_backend="native", libcint_geometry_grad_policy="analytic",
    precompile_eri=False, precompile_eri_chunk_size=512, _precompile_eri_kernels=precompile_eri_kernels,
    include_dipole_integrals=True, init_guess="hcore", chkfile=None,
    init_guess_sap_basis=None, init_guess_chkfile_project=None, verbose=0, **mol_kwargs):
    from gradscf.scf.rks import RKSConfig
    from gradscf.scf.init_guess import restricted_initial_guess
    cfg, name = _resolve_config(config, xc_spec, RKSConfig)
    actual_spin = atom.spin if isinstance(atom, MoleculeSpec) else spin
    if actual_spin != 0:
        raise NotImplementedError("Restricted inputs require a closed-shell system")
    values = _build_common(atom=atom,basis=basis,cfg=cfg,xc_spec=name,unit=unit,charge=charge,spin=spin,
        cart=cart,grids_level=grids_level,max_l=max_l,grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,geometry_grad_policy=libcint_geometry_grad_policy,
        include_dipole=include_dipole_integrals,precompile_eri=precompile_eri,
        precompile_eri_chunk_size=precompile_eri_chunk_size,precompiler=_precompile_eri_kernels,
        chkfile=chkfile,sap_basis=init_guess_sap_basis,chkfile_project=init_guess_chkfile_project,mol_kwargs=mol_kwargs)
    spec,ctx,s,h,eri,pair,factors,dipole,ao,dao,lap,mode,eri_lr,dispersion_energy=values
    initial = restricted_initial_guess(init_guess=init_guess,dtype=h.dtype)
    nelectron = int(np.asarray(spec.charges).sum())-int(spec.charge)
    direct=None
    if cfg.jk_backend=='direct':
        direct=ctx.basis
        if mode=='native':
            from gradscf.integrals.backends.native.jk import NativeDirectBasis
            top,parameters=prepare_basis(spec,basis,cart=True)
            direct=NativeDirectBasis(make_plan(top),parameters)
    return RKSIntegralInputs(basis=ctx.basis,overlap=s,hcore=h,eri=eri if eri_lr is not None else None,
        eri_pair_matrix=pair if cfg.jk_backend == "full" else None,df_factors=factors,
        direct_basis=direct,nelectron=nelectron,
        nuclear_repulsion=spec.nuclear_repulsion,coords=ctx.coords,grid_weights=ctx.grid_weights,
        ao=ao,ao_deriv1=dao,ao_laplacian=lap,dipole_integrals=dipole,init_density=initial.density,
        molecule_charge=int(spec.charge),geometry_is_traced=ctx.geometry_is_traced,
        integral_backend=mode,grid_ao_backend="jax",eri_lr=eri_lr,dispersion_energy=dispersion_energy)


def build_uks_integral_inputs(*, atom, basis, config=None, xc_spec=None,
    unit="Angstrom", charge=0, spin=1, cart=True, grids_level=0, max_l=3,
    grid_ao_backend="jax", integral_backend="native", libcint_geometry_grad_policy="analytic",
    precompile_eri=False, precompile_eri_chunk_size=512, _precompile_eri_kernels=precompile_eri_kernels,
    init_guess="hcore", chkfile=None, init_guess_sap_basis=None,
    init_guess_chkfile_project=None, verbose=0, **mol_kwargs):
    from gradscf.scf.uks import UKSConfig
    from gradscf.scf.init_guess import unrestricted_initial_guess
    cfg, name = _resolve_config(config, xc_spec, UKSConfig)
    values = _build_common(atom=atom,basis=basis,cfg=cfg,xc_spec=name,unit=unit,charge=charge,spin=spin,
        cart=cart,grids_level=grids_level,max_l=max_l,grid_ao_backend=grid_ao_backend,
        integral_backend=integral_backend,geometry_grad_policy=libcint_geometry_grad_policy,
        include_dipole=True,precompile_eri=precompile_eri,precompile_eri_chunk_size=precompile_eri_chunk_size,
        precompiler=_precompile_eri_kernels,chkfile=chkfile,sap_basis=init_guess_sap_basis,
        chkfile_project=init_guess_chkfile_project,mol_kwargs=mol_kwargs)
    spec,ctx,s,h,eri,pair,factors,dipole,ao,dao,lap,mode,eri_lr,dispersion_energy=values
    initial=unrestricted_initial_guess(init_guess=init_guess,dtype=h.dtype)
    total=int(np.asarray(spec.charges).sum())-int(spec.charge)
    nalpha,nbeta=_unrestricted_spin_electron_counts(total,int(spec.spin))
    direct=None
    if cfg.jk_backend == "direct":
        if mode != "native":
            raise NotImplementedError("Unrestricted direct SCF requires native AO J/K.")
        from gradscf.integrals.backends.native.jk import NativeDirectBasis
        top,parameters=prepare_basis(spec,basis,cart=True)
        direct=NativeDirectBasis(make_plan(top),parameters)
        eri=jnp.zeros((0,0,0,0),dtype=h.dtype)
    elif cfg.jk_backend == "df":
        eri=jnp.zeros((0,0,0,0),dtype=h.dtype)
    elif mode=='native' and eri_lr is None:
        eri=pair
    elif eri is None:
        eri=eri_tensor(ctx.basis)
    return UKSIntegralInputs(basis=ctx.basis,overlap=s,hcore=h,eri=eri,df_factors=factors,direct_basis=direct,
        nalpha=nalpha,nbeta=nbeta,nuclear_repulsion=spec.nuclear_repulsion,
        coords=ctx.coords,grid_weights=ctx.grid_weights,ao=ao,ao_deriv1=dao,ao_laplacian=lap,
        dipole_integrals=dipole,init_density_alpha=initial.density_alpha,init_density_beta=initial.density_beta,
        total_electrons=total,molecule_charge=int(spec.charge),geometry_is_traced=ctx.geometry_is_traced,
        integral_backend=mode,grid_ao_backend="jax",eri_lr=eri_lr,dispersion_energy=dispersion_energy)

__all__ = ["RKSIntegralInputs", "UKSIntegralInputs", "build_rks_integral_inputs", "build_uks_integral_inputs"]
