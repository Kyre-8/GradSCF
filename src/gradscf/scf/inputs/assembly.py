"""Standalone integral/grid assembly for restricted and unrestricted SCF."""
from __future__ import annotations

from dataclasses import replace
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np

from gradscf._memory import DEFAULT_MAX_MEMORY_MB, workspace_bytes
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


def _eri_workspace_bytes(nao, cfg, *, mode, rsh, geometry_is_traced,
                         itemsize=8, auxiliary_nao=None):
    """Forecast known integral/factor arrays from dimensions, not values."""
    npair = nao * (nao + 1) // 2
    if mode == "native" and auxiliary_nao is not None:
        # Metric and its factorization, raw/whitened three-center arrays,
        # and the unpacked factors can coexist. The retained rank is not yet
        # known, so use the complete auxiliary dimension without truncating it.
        return itemsize * (2*auxiliary_nao**2 + 2*auxiliary_nao*npair
                           + auxiliary_nao*nao**2)
    if mode == "native" and rsh is not None:
        return 2 * nao**4 * itemsize
    required = (nao**4 + npair**2 if mode == "native" and geometry_is_traced
                else npair**2) * itemsize
    if cfg.jk_backend == "df":
        rank_cap = getattr(cfg, "df_max_rank", None)
        rank = min(npair, int(rank_cap)) if rank_cap is not None and int(rank_cap) > 0 else npair
        # A rank cap limits factor outputs, but spectral factorization still
        # constructs the symmetric pair matrix and its full eigenvectors.
        required += itemsize * (2*npair**2 + npair + rank*(npair + nao**2))
    return required


def _check_eri_memory(context, cfg, *, mode, rsh, needs_ao_laplacian,
                       scf_history_matrices, required_bytes=None):
    """Reject a known oversized integral representation before evaluating it."""
    if required_bytes is None:
        if cfg.jk_backend == "direct":
            return
        required_bytes = _eri_workspace_bytes(context.basis.nao, cfg, mode=mode, rsh=rsh,
            geometry_is_traced=context.geometry_is_traced,
            itemsize=max(8, np.dtype(context.coords.dtype).itemsize))
    bundle = context.grid_ao_bundle
    arrays = () if bundle is None else (bundle.ao, bundle.ao_deriv1, bundle.ao_laplacian)
    resident = _grid._grid_resident_bytes(context.basis, context.coords,
        context.grid_weights, *arrays, scf_max_cycle=cfg.max_cycle,
        scf_history_matrices=scf_history_matrices)
    device = _grid._grid_device(context.coords)
    available = workspace_bytes(getattr(cfg, "max_memory", DEFAULT_MAX_MEMORY_MB),
                                resident_bytes=resident, device=device)
    provider = context.grid_ao
    ao_workspace = (0 if provider is None else provider.block_size *
                    _grid.ao_point_bytes(provider, deriv=2 if needs_ao_laplacian else 1))
    if required_bytes + ao_workspace > available:
        remedy = ("reduce direct_scf_workers or increase max_memory."
                  if cfg.jk_backend == "direct" else "use jk_backend='direct' or increase max_memory.")
        raise ValueError(
            f"The integral representation/workspace needs at least {required_bytes / 1e6:.1f} MB, "
            f"plus {ao_workspace / 1e6:.1f} MB of AO workspace, exceeding "
            f"the remaining max_memory budget ({available / 1e6:.1f} MB); {remedy}"
        )


def _build_common(*, atom, basis, cfg, xc_spec, unit, charge, spin, cart,
                  grids_level, max_l, grid_ao_backend, integral_backend,
                  geometry_grad_policy, include_dipole, precompile_eri,
                  precompile_eri_chunk_size, precompiler, chkfile, sap_basis,
                  chkfile_project, mol_kwargs, scf_history_matrices=2):
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
    fixed_workspace = 0
    auxiliary = getattr(cfg, "auxbasis", None)
    if mode == "native":
        # Preparing metadata is cheap; forecast native/RI storage before the
        # grid planner commits the remaining budget to AO work arrays.
        top, params = prepare_basis(spec, basis, cart=True)
        if geometry_grad_policy == "zero":
            params = replace(params, centers=jax.lax.stop_gradient(params.centers),
                             nuclear_coords=jax.lax.stop_gradient(params.nuclear_coords))
        plan = make_plan(top, backend="native")
        geometry_is_traced = any(isinstance(value, jax.core.Tracer)
            for value in jax.tree_util.tree_leaves(spec.coords_bohr))
        auxiliary_nao = None
        if cfg.jk_backend == "df" and auxiliary is not None:
            if geometry_is_traced:
                raise NotImplementedError('Native auxiliary-integral geometry AD is not implemented.')
            aux_top, aux_params = prepare_basis(spec, auxiliary, cart=True)
            auxiliary_nao = aux_top.nao
        if cfg.jk_backend == "direct":
            from gradscf.integrals.backends.native.jk import NativeDirectBasis, native_jk_workspace_bytes

            estimate = native_jk_workspace_bytes(NativeDirectBasis(plan, params),
                                                 cfg.direct_scf_workers, cfg.direct_scf_tol)
            if estimate is None:
                raise ValueError("Native J/K workspace requires a complete basis topology.")
            fixed_workspace = 3 * estimate
        else:
            fixed_workspace = _eri_workspace_bytes(top.nao, cfg, mode=mode, rsh=rsh,
                geometry_is_traced=geometry_is_traced, auxiliary_nao=auxiliary_nao)
    context = _prepare_basis_grid_context(spec=spec, basis=basis, max_l=max_l,
        grids_level=grids_level, precompute_eri_groups=(mode == "jax"),
        needs_ao_laplacian=laplacian, needs_grid=needs_grid,
        grid_block_size=getattr(cfg, "grid_block_size", -1),
        grid_max_memory=getattr(cfg, "grid_max_memory", None),
        max_memory=getattr(cfg, "max_memory", DEFAULT_MAX_MEMORY_MB),
        scf_max_cycle=cfg.max_cycle, scf_history_matrices=scf_history_matrices,
        fixed_workspace_bytes=fixed_workspace)
    if context.geometry_is_traced and geometry_grad_policy == "error":
        raise NotImplementedError("Geometry gradients are disabled by geometry_grad_policy='error'")
    _check_eri_memory(context, cfg, mode=mode, rsh=rsh,
                      needs_ao_laplacian=laplacian,
                      scf_history_matrices=scf_history_matrices,
                      required_bytes=fixed_workspace if mode == "native" else None)
    eri_lr = dispersion_energy = None
    if original_chg:
        from gradscf.dft.dispersion import wb97xd_dispersion_energy
        dispersion_energy = wb97xd_dispersion_energy(spec.symbols, spec.coords_bohr)
    if mode == "native":
        s = plan.evaluate("overlap", params)
        h = plan.evaluate("kinetic", params)+plan.evaluate("nuclear", params)
        full_eri=pair=factors=None
        if cfg.jk_backend=='direct':
            if context.geometry_is_traced and cfg.direct_scf_tol != 0.:
                raise ValueError('Direct-SCF geometry AD requires direct_scf_tol=0.')
        elif rsh is not None:
            full_eri = plan.evaluate("eri", params)
            eri_lr = plan.evaluate("eri", params, omega=rsh.default_omega)
        elif cfg.jk_backend=='df' and auxiliary is not None:
            from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan, unpack_factors
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
        chkfile=chkfile,sap_basis=init_guess_sap_basis,chkfile_project=init_guess_chkfile_project,
        mol_kwargs=mol_kwargs)
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
        integral_backend=mode,grid_ao_backend="jax",eri_lr=eri_lr,dispersion_energy=dispersion_energy,
        grid_ao=ctx.grid_ao)


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
        chkfile_project=init_guess_chkfile_project,mol_kwargs=mol_kwargs,
        scf_history_matrices=6)
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
        integral_backend=mode,grid_ao_backend="jax",eri_lr=eri_lr,dispersion_energy=dispersion_energy,
        grid_ao=ctx.grid_ao)

__all__ = ["RKSIntegralInputs", "UKSIntegralInputs", "build_rks_integral_inputs", "build_uks_integral_inputs"]
