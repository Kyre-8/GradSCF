"""Independent full-space or symmetry-packed RS references for examples only.

Uses PySCF integrals/actions and NumPy recurrence, without GradSCF or Taylor AD.
``frozen`` is a nonnegative core count applied to both spin channels. This
``mp_coefficients`` allocates full determinant vectors, so use small systems.
``symmetry_reference`` folds the core and packs the complete spatial symmetry
sector; it does not truncate virtual orbitals or excitation ranks.
"""
import numpy as np
from pyscf import ao2mo, fci


def mp_coefficients(mf, order, frozen=0):
    """Return individual E2 through Eorder in Hartree for a converged RHF/UHF."""
    unrestricted = np.asarray(mf.mo_coeff).ndim == 3
    coeff = mf.mo_coeff if unrestricted else (mf.mo_coeff, mf.mo_coeff)
    ca, cb = coeff
    nmo = ca.shape[1]
    na, nb = mf.mol.nelec
    if not mf.converged or not 0 <= frozen <= min(na, nb):
        raise ValueError("Use a converged reference and a valid frozen-core count")
    h = tuple(c.T @ mf.get_hcore() @ c for c in (coeff if unrestricted else (ca,)))
    frames = ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb)) if unrestricted else ((ca,)*4,)
    g = tuple(ao2mo.general(mf.mol, cs, compact=False).reshape((nmo,)*4)
              for cs in frames)
    ea, eb = mf.mo_energy if unrestricted else (mf.mo_energy, mf.mo_energy)
    aocc = fci.cistring.gen_occslst(range(nmo), na)
    bocc = fci.cistring.gen_occslst(range(nmo), nb)
    phi = np.zeros((len(aocc), len(bocc)))
    phi[0, 0] = 1.
    engine = fci.direct_uhf if unrestricted else fci.direct_spin1
    tensor = engine.absorb_h1e(h if unrestricted else h[0],
        g if unrestricted else g[0], nmo, (na, nb), .5)
    action = lambda c: np.asarray(engine.contract_2e(tensor, c, nmo, (na, nb)))
    gaps = (np.sum(ea[aocc], axis=1)[:, None]+np.sum(eb[bocc], axis=1)[None, :]
            -np.sum(ea[:na])-np.sum(eb[:nb]))
    amask = np.array([all(i in occ for i in range(frozen)) for occ in aocc])
    bmask = np.array([all(i in occ for i in range(frozen)) for occ in bocc])
    mask = amask[:, None] & bmask[None, :]
    return rs_coefficients(action, phi, gaps, order, mask=mask)


def rs_coefficients(action, phi, gaps, order, *, mask=None, callback=None):
    """Independent NumPy RS recurrence, also accepting symmetry-packed vectors."""
    reference = int(np.argmax(np.asarray(phi).reshape(-1)))
    e0 = action(phi).reshape(-1)[reference]
    mask = np.ones(phi.shape, dtype=bool) if mask is None else mask.copy()
    mask.reshape(-1)[reference] = False
    if np.any(np.abs(gaps[mask]) < 1e-12):
        raise ValueError("The reference is not isolated in the retained space")
    waves, energies = [phi], [e0]
    for degree in range(1, order+1):
        source = np.asarray(action(waves[-1]))-(e0+gaps)*waves[-1]
        energies.append(source.reshape(-1)[reference])
        for j in range(1, degree):
            source -= energies[j]*waves[degree-j]
        if degree < order:
            waves.append(np.where(mask, -source/np.where(mask, gaps, 1.), 0.))
        if callback is not None:
            callback(degree, float(energies[-1]))
    return np.asarray(energies[2:])


def symmetry_reference(mf, frozen=0):
    """Frozen-core RHF Hamiltonian in PySCF's packed Ag/D2h string blocks.

    No virtual orbitals or excitation ranks are removed. This is an external
    NumPy/PySCF benchmark adapter, not a GradSCF differentiable backend.
    """
    from pyscf import mcscf
    from pyscf.fci import direct_spin1_symm
    if not mf.converged or mf.mol.groupname != "D2h" or np.ndim(mf.mo_coeff) != 2:
        raise ValueError("Use a converged D2h RHF reference")
    norb = mf.mo_coeff.shape[1]-frozen
    nelec = tuple(n-frozen for n in mf.mol.nelec)
    if nelec[0] != nelec[1] or min(nelec) < 1:
        raise ValueError("This benchmark adapter requires a nonempty closed shell")
    cas = mcscf.CASCI(mf,norb,nelec)
    h,ecore = cas.get_h1eff()
    g = ao2mo.restore(1,cas.get_h2eff(),norb)
    irreps = np.asarray(mf.mo_coeff.orbsym[frozen:],dtype=np.int32) % 10
    occupied = fci.cistring.gen_occslst(range(norb),nelec[0])
    string_symmetry = np.bitwise_xor.reduce(irreps[occupied],axis=1)
    groups = [np.flatnonzero(string_symmetry == ir) for ir in range(8)]
    eps = np.asarray(mf.mo_energy[frozen:])
    excitation = eps[occupied].sum(axis=1)-eps[:nelec[0]].sum()
    gaps = np.concatenate([(excitation[ix,None]+excitation[ix][None,:]).ravel()
                           for ix in groups])
    phi = np.zeros_like(gaps)
    reference_symmetry = string_symmetry[0]
    offset = sum(len(ix)**2 for ix in groups[:reference_symmetry])
    reference_pos = int(np.flatnonzero(groups[reference_symmetry] == 0)[0])
    phi[offset+reference_pos*(len(groups[reference_symmetry])+1)] = 1.
    links = fci.cistring.gen_linkstr_index(range(norb),nelec[0],tril=True)
    effective = fci.direct_spin1.absorb_h1e(h,g,norb,nelec,.5)
    def action(vector):
        return np.asarray(direct_spin1_symm.contract_2e(effective,vector,norb,nelec,
            link_index=(links,links),orbsym=irreps,wfnsym=0))
    occupations = np.zeros((len(occupied),norb))
    np.put_along_axis(occupations,occupied,1.,axis=1)
    coulomb,exchange = np.einsum("ppqq->pq",g),np.einsum("pqqp->pq",g)
    same = occupations @ np.diag(h)+.5*np.einsum(
        "pi,ij,pj->p",occupations,coulomb-exchange,occupations)
    diagonal = np.concatenate([(same[ix,None]+same[ix][None,:]
        +occupations[ix] @ coulomb @ occupations[ix].T).ravel() for ix in groups])
    return action,phi,gaps,diagonal,float(ecore)
