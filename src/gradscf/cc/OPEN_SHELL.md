# Open-shell and unrestricted post-HF

## Implemented scope

This is a real, collinear, molecular, in-core reference implementation. UHF
supplies separate alpha/beta orbitals. ROHF supplies the same spatial orbitals
for both spins, with separate occupied spaces. The equations conserve N-alpha
and N-beta; they do not project total spin S.

| API | Implemented meaning |
| --- | --- |
| `ci.UCI`, `UCISD`, `UCISDT`, `UCISDTQ` | Determinant CI through the requested excitation rank, relative to a UHF or ROHF determinant |
| `ci.UCIS` | Spin-conserving singles excitation energies for a stationary UHF reference; equivalent to UHF/TDA |
| `cc.UCCSD`, `cc.UCCD` | Spin-conserving spin-orbital CCSD or doubles-only projected CC equations |
| `ci.CISD`, `CISDT`, `CISDTQ`, `CI`, `cc.CCSD`, `cc.CCD` | Dispatch from the reference to restricted or unrestricted equations |
| `cc.RCCSD` | Explicitly restricted closed-shell input |

ROHF-based UCCSD here means unrestricted cluster amplitudes on ROHF orbitals.
It is **not** an implementation of a distinct spin-adapted ROCCSD formulation.
ROHF orbitals are not generally stationary for independent alpha/beta orbital
rotations, so `UCIS` rejects nonzero spin Fock occupied-virtual blocks. Variational
rank-truncated CI and UCCSD can use these noncanonical reference orbitals.

Unrestricted/ROHF QCISD/(T), unrestricted CC2, LCCSD, LCCD, spin-flip CIS, complex orbitals and spin-mixed
GHF are not exposed as implemented methods. `CIS_D` remains restricted.

UCCSD/UCCD now expose Lambda and MO 1/2-RDMs, including spin-dependent frozen
orbitals. `ccsd_t()` / `triples_correction` also accept canonical real UHF CCSD
states. The entire active Fock matrix must be diagonal within `canonical_tol`;
ordinary ROHF or rotated noncanonical inputs are rejected by this default path.
The explicit `orbital_basis="semicanonical"` option now evaluates the
general-reference `(T)` expression, including `F_vo*T2`, via a tensor resolvent.
It is equivalent to transforming amplitudes/integrals consistently into
semicanonical occupied/virtual frames and is differentiable at within-block
degeneracies through the shared implicit linear solve. Its six-index memory cap
and conservative full-spectrum denominator policy are described in
[SEMICANONICAL.md](SEMICANONICAL.md). Only the conventional `(T)` variant is
exposed for unrestricted inputs. No CCSD(T) density or nuclear-gradient facade
is claimed, nor a separately spin-adapted ROCCSD method.

## Hamiltonian and coordinates

The shared `scf.reference.UnrestrictedReference` stores

```text
h1 = (ha, hb)
eri = (gaa, gab, gbb)
nocc = (nalpha, nbeta)
gab[p,q,r,s] = (p_alpha q_alpha | r_beta s_beta)
```

Each spin orbital frame is orthonormal, real, equally sized, and ordered occupied
before virtual. All energies and denominators are in Hartree. Full AO ERIs,
s4 pair matrices and density-fitted AO factors can be transformed through
`ao2mo.transform_unrestricted_integrals`. Mixed-spin ERIs use both orbital
frames. They are not obtained by reusing a restricted spatial tensor.

The chemists' spin-orbital tensor inserts spin deltas on each orbital pair.
Alpha orbitals precede beta orbitals in CI determinants. Slater-Condon connections
are shared with restricted CI; spin-resolved integrals retain their full orbital
indices. A rank-k space includes the reference and excitations with
`rank_alpha + rank_beta <= k`. Frozen occupied electrons remain in every
determinant, preserving their constant energy and interaction with active electrons.

CC uses occupied-alpha, occupied-beta, virtual-alpha, virtual-beta ordering and
antisymmetrized physicists' integrals

\[
v_{pqrs}=\langle pq\Vert rs\rangle=(pr|qs)-(ps|qr),\qquad
F_{pq}=h_{pq}+\sum_{i\in\mathrm{all\ occupied}}\langle pi\Vert qi\rangle.
\]

All occupied electrons contribute before frozen orbitals are removed. CCSD solves

\[
R_\mu(T)=\langle\Phi_\mu|e^{-T}He^T|\Phi_0\rangle=0,\quad
T=T_1+T_2,
\]

with correlation energy

\[
E_c=\sum_{ia}F_{ia}t_i^a+
\frac14\sum_{ijab}v_{ijab}t_{ij}^{ab}+
\frac12\sum_{ijab}v_{ijab}t_i^at_j^b.
\]

Only spin-conserving singles and doubles with `i<j`, `a<b` are independent.
Each independent double is packed with factor 2 so the packed Euclidean norm
equals the full tensor norm. Unpacking restores both fermionic antisymmetries.
This removes redundant coordinates from the implicit Jacobian. UCCD holds
singles at zero and projects only doubles residuals.

Public amplitudes follow the PySCF UCCSD layout:

```text
t1 = (ta[oa,va], tb[ob,vb])
t2 = (taa[oa,oa,va,va], tab[oa,ob,va,vb], tbb[ob,ob,vb,vb])
```

`frozen=1` freezes the first occupied orbital in each channel. A flat list
freezes the same spatial indices in both channels; `frozen=([0], [0, 5])`
selects separate alpha/beta frozen sets. Frozen virtuals remain empty. Amplitude
shapes describe the resulting active spaces.

## Forward and backward ownership

CI calls `solvers.solve_hermitian`; CC calls
`solvers.nonlinear.solve_nonlinear`. There are no additional eigensolver,
DIIS, Newton or adjoint iteration implementations in the method modules.
Level shifts modify only the CC preconditioner, not the physical residual.

For external integral parameters theta, implicit response satisfies

\[
R_T\,dT=-R_\theta\,d\theta.
\]

Reverse mode uses the corresponding transposed linear equation in the common
solver. Energy and amplitude observables include converged-amplitude response.
CI coefficient response requires `gradient_mode="implicit_eigenvector"` and
an isolated root. Unconverged roots invalidate response rather than silently
returning zero gradients.

The verified contract is first-order differentiation of fixed-topology MO
integrals. Orbital response and nuclear gradients require differentiable upstream
SCF/integrals and are not established by these fixed-MO tests. The eager facade
validates convergence and SCF input freshness; use `run_ucc`/`solve_ci` under JIT.

## Cost and references

The initial implementation materializes dense spin ERIs with O((2*nmo)^4)
storage. It supports DF input transformation but is not a DF-CC contraction
algorithm. CI retains the existing determinant and connection limits.

The CC contractions are adapted from PySCF 2.9.0 `gccsd` and `gintermediates`;
see [NOTICE.md](NOTICE.md) for exact source hashes and Apache-2.0 attribution.
The upstream intermediates cite Gauss and Stanton, J. Chem. Phys. 103, 3561
(1995), Table III, [doi:10.1063/1.470240](https://doi.org/10.1063/1.470240).
This records the upstream working-equation attribution; it does not claim an
independent full-text audit or implement NMR properties.
General CC theory is cited in [REFERENCES.md](REFERENCES.md); determinant CI
theory and the distinction between method and software citations are in
[CI references](../ci/REFERENCES.md). No OpenMolcas source was copied.
