"""PySCF object adapters confined to independent comparison tests."""
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from gradscf.integrals.basis import CartesianAO, CartesianBasis, ContractedShell, cartesian_angular_tuples
from gradscf.dft.hfx import ChunkedHFXNu

def _int1e_grids_name(mol: Any) -> str:
    return "int1e_grids_cart" if bool(getattr(mol, "cart", False)) else "int1e_grids_sph"


def _int1e_rinv_name(mol: Any) -> str:
    return "int1e_rinv_cart" if bool(getattr(mol, "cart", False)) else "int1e_rinv_sph"


def build_chunked_hfx_nu_from_mol(
    mol: Any,
    coords: Any,
    *,
    omega_values: tuple[float, ...],
    nao: int,
    chunk_size: int = 512,
) -> "ChunkedHFXNu":
    coords_arr = np.asarray(coords)
    omega_values = tuple(float(omega) for omega in omega_values)
    shape = (
        len(omega_values),
        int(coords_arr.shape[0]),
        int(nao),
        int(nao),
    )
    int1e_grids = _int1e_grids_name(mol)
    int1e_rinv = _int1e_rinv_name(mol)

    def grid_chunk(start: int, stop: int) -> np.ndarray:
        start_i = int(start)
        stop_i = int(stop)
        coords_chunk = coords_arr[start_i:stop_i]
        chunks = []
        for omega in omega_values:
            try:
                with mol.with_range_coulomb(omega=float(omega)):
                    nu = mol.intor(int1e_grids, hermi=1, grids=coords_chunk)
            except TypeError:
                nu_list = []
                with mol.with_rinv_zeta(zeta=float(omega) * float(omega)):
                    for coord in coords_chunk:
                        with mol.with_rinv_origin(coord):
                            nu_list.append(mol.intor(int1e_rinv, hermi=1))
                nu = np.asarray(nu_list)
            chunks.append(np.asarray(nu))
        return np.stack(chunks, axis=0)

    callback_dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.float64))

    def grid_chunk_padded(start: Any, chunk_size: int) -> jnp.ndarray:
        chunk_size_i = int(chunk_size)
        chunk_shape = (shape[0], chunk_size_i, shape[2], shape[3])

        def read_chunk(start_arg: Any) -> np.ndarray:
            read_start = int(np.asarray(start_arg))
            read_stop = min(read_start + chunk_size_i, shape[1])
            output = np.zeros(chunk_shape, dtype=callback_dtype)
            if read_stop <= read_start:
                return output
            data = np.asarray(grid_chunk(read_start, read_stop), dtype=callback_dtype)
            output[:, : data.shape[1]] = data
            return output

        return jax.pure_callback(
            read_chunk,
            jax.ShapeDtypeStruct(chunk_shape, callback_dtype),
            jnp.asarray(start, dtype=jnp.int32),
        )

    return ChunkedHFXNu(
        shape=shape,
        chunk_size=int(chunk_size),
        _grid_chunk_fn=grid_chunk,
        _grid_chunk_padded_fn=grid_chunk_padded,
    )


def _local_hfx_features_from_dm(
    mol: Any,
    ao: np.ndarray,
    dm_spin: tuple[np.ndarray, np.ndarray],
    coords: np.ndarray,
    *,
    omega_values: tuple[float, ...],
    chunk_size: int = 512,
    return_nu: bool = False,
    return_fxx: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Compute molecule-local HF exchange channels used by neural functionals."""

    dm_a, dm_b = dm_spin
    e_a = ao @ dm_a
    e_b = ao @ dm_b
    ngrid = int(coords.shape[0])
    n_omega = len(omega_values)
    nao = int(ao.shape[1])
    hfx = np.zeros((2, ngrid, n_omega), dtype=np.float64)
    nu_cache = (
        np.zeros((n_omega, ngrid, nao, nao), dtype=np.float64) if return_nu else None
    )
    fxx_cache = (
        np.zeros((n_omega, ngrid, nao), dtype=np.float64) if return_fxx else None
    )
    int1e_grids = _int1e_grids_name(mol)
    int1e_rinv = _int1e_rinv_name(mol)

    for omega_idx, omega in enumerate(omega_values):
        for start in range(0, ngrid, int(chunk_size)):
            end = min(start + int(chunk_size), ngrid)
            coords_chunk = coords[start:end]
            try:
                with mol.with_range_coulomb(omega=float(omega)):
                    nu = mol.intor(int1e_grids, hermi=1, grids=coords_chunk)
            except TypeError:
                nu_list = []
                with mol.with_rinv_zeta(zeta=float(omega) * float(omega)):
                    for coord in coords_chunk:
                        with mol.with_rinv_origin(coord):
                            nu_list.append(mol.intor(int1e_rinv, hermi=1))
                nu = np.asarray(nu_list)
            if nu_cache is not None:
                nu_cache[omega_idx, start:end] = nu

            e_a_chunk = e_a[start:end]
            e_b_chunk = e_b[start:end]
            fxx_a = np.einsum("gbc,gc->gb", nu, e_a_chunk, optimize=True)
            fxx_b = np.einsum("gbc,gc->gb", nu, e_b_chunk, optimize=True)
            if fxx_cache is not None:
                fxx_cache[omega_idx, start:end] = 0.5 * (fxx_a + fxx_b)
            hfx[0, start:end, omega_idx] = -0.5 * np.einsum(
                "gb,gb->g", e_a_chunk, fxx_a, optimize=True
            )
            hfx[1, start:end, omega_idx] = -0.5 * np.einsum(
                "gb,gb->g", e_b_chunk, fxx_b, optimize=True
            )
    outputs: list[np.ndarray] = [hfx]
    if nu_cache is not None:
        outputs.append(nu_cache)
    if fxx_cache is not None:
        outputs.append(fxx_cache)
    if len(outputs) == 1:
        return hfx
    return tuple(outputs)


def basis_from_pyscf_mol_cart(
    mol: Any,
    *,
    max_l: int = 3,
    precompute_eri_groups: bool = True,
) -> CartesianBasis:
    """Build a Cartesian AO basis from a PySCF Mole.

    Notes:
    - Requires `mol.cart = True` for direct cartesian AO ordering.
    - Current integral engine supports up to `l=3` (s/p/d/f).
    """

    if not bool(getattr(mol, "cart", False)):
        raise ValueError(
            "basis_from_pyscf_mol_cart requires a PySCF Mole with cart=True."
        )

    aos: list[CartesianAO] = []
    shells: list[ContractedShell] = []
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        if l > max_l:
            raise NotImplementedError(
                f"Current JAX integral implementation supports l<= {max_l}, got l={l}."
            )
        atom_idx = int(mol.bas_atom(ib))
        center = np.asarray(mol.atom_coord(atom_idx), dtype=float)
        exponents = np.asarray(mol.bas_exp(ib), dtype=float)
        ctr_coeff = np.asarray(mol.bas_ctr_coeff(ib), dtype=float)  # (nprim, nctr)
        if ctr_coeff.ndim == 1:
            ctr_coeff = ctr_coeff[:, None]

        for ctr in range(ctr_coeff.shape[1]):
            coeff = ctr_coeff[:, ctr]
            angulars = tuple(cartesian_angular_tuples(l))
            shell_start = len(aos)
            for angular in angulars:
                aos.append(
                    CartesianAO(
                        center=center,
                        angular=angular,
                        exponents=exponents,
                        coefficients=coeff,
                    )
                )
            shell_stop = len(aos)
            shells.append(
                ContractedShell(
                    center=center,
                    angulars=angulars,
                    exponents=exponents,
                    coefficients=coeff,
                    ao_indices=np.arange(shell_start, shell_stop, dtype=np.int32),
                )
            )

    return CartesianBasis(
        aos=tuple(aos),
        precompute_eri_groups=bool(precompute_eri_groups),
        atom_coords=np.asarray(mol.atom_coords(), dtype=float),
        atom_charges=np.asarray(mol.atom_charges(), dtype=float),
        shells=tuple(shells),
    )
