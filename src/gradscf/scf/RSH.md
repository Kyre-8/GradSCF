# Fixed original omegaB97X-D molecular SCF

The molecular RKS/UKS full-ERI path recognizes the existing `wb97xd`,
`wb97x-d`, and `wb97x_d` preset aliases. Its fixed parameters are omega =
0.2 Bohr^-1, short-range HF fraction 0.222036, and long-range fraction 1.
This is the original functional with Chai-Head-Gordon dispersion, not D3/D4.

The shared energy/Fock algebra uses the ordinary Coulomb J and

    K_effective = 0.222036 K_full + 0.777964 K_long_range(omega=0.2).

The LibXC-compatible local channel is `hyb_gga_xc_wb97x_d`. That channel alone
contains neither exact exchange nor CHG dispersion and is not the complete
method. `parse_xc` remains a local/global-XC parser; complete-method dispatch is
owned by the internal `scf.rsh` helper and molecular SCF assembly.

For this local channel only, both SCF spin paths omit quadrature points whose
total density is at or below `config.density_floor` (default 1e-12 electrons
per Bohr^3). Inactive inputs are replaced by a regular dummy point before
evaluating the local XC expression, then their energy and potential are
masked consistently. The active local expression is unchanged. This is an
explicit numerical integration cutoff, with a nonsmooth boundary at the
threshold; it is not an assertion that the untruncated generated expression
has finite high-order derivatives at every density. Smaller cutoffs require
new derivative checks. Merely having finite energy/potential is insufficient.

On the fixed-density STO-3G/level-0 reference grids used here, the default
cutoff removes 45 water and 63 OH points, whose summed XC contributions are
9.61e-16 and 3.12e-15 Hartree, respectively. Refining to 1e-13 changes the
converged total energies by -4.26e-14 and 0 Hartree (floating-point resolution),
but the original expanded jax_xc expression retained 2 water and 3 OH points
with nonfinite local Hessians. Response validation uses the default 1e-12
threshold; energy insensitivity does not validate smaller-cutoff derivatives.

The private `dft.libxc_jax.wb97_local` kernel evaluates the same B97 expansion
as a Horner polynomial in the bounded variable `u = gamma*s^2/(1+gamma*s^2)`.
Its exchange and same/opposite-spin correlation follow Eqs. 6--7 and 11--26
of [Chai and Head-Gordon, JCP 128, 084106](https://doi.org/10.1063/1.2834918),
with the original omegaB97X-D factory coefficients. It preserves the PW92
parameter rounding (`fz20 = 1.709921`) and the LibXC short-range attenuation
branch/series. Exactly at `a = 1.35`, AD uses the selected series branch's
right derivative; it does not reproduce the upstream nested-clamp artifact
that gave zero there. The finite series is not forced to be infinitely smooth
across this existing approximation boundary. This algebraic factorization avoids the high density-reciprocal
powers in the generated expression; it neither raises the integration cutoff
nor replaces nonfinite derivatives. Before factorization, the default-cutoff
water reference grid still had 36 nonfinite third-derivative points at total
densities 1.22e-12 to 1.75e-10; OH had 40 at 1.16e-12 to 3.87e-10. Thus finite
Hessians alone were insufficient for the explicit orbital Newton trajectory,
whose derivatives also involve third local derivatives.

At total density about 1e-11, LibXC 7's expanded double-precision kernel also
loses accuracy in some small entries despite a finite result. The tail test
therefore checks a small Hessian element against the original expanded
expression evaluated with 80-digit mpmath arithmetic, while retaining the
full-kernel LibXC comparison at its numerical scale. The reproducible script
`tests/comparisons/wb97xd_tail_high_precision.py` records the jax_xc 0.0.9
source SHA256, the corrected three B97 coefficient vectors, and PW92 rounding.
Ordinary-density LibXC energy/potential/kernel checks retain their strict
elementwise tolerances.

`build_rks_integral_inputs` / `build_uks_integral_inputs` construct ordinary
full ERIs, `eri_lr`, and `dispersion_energy` from native integrals and the
geometry-only CHG expression. CHG currently supports H/C/N/O/F. Both ERI
tensors occupy O(nao^4) storage. Cartesian molecular grids/AOs and the native
CPU float64 backend are required; DF and direct-SCF are explicitly rejected.

Low-level `run_rks_from_integrals`, its traceable variant, and
`run_uks_from_integrals` require explicit `eri_lr` and `dispersion_energy` when
the complete method name is selected. Their defaults are `None` so a missing
component is distinguishable from a valid zero dispersion correction.
`total_energy = electronic_energy + nuclear_repulsion + dispersion_energy`;
the nuclear-repulsion field continues to contain only the nuclear Coulomb term.
For RSH results, `exact_exchange_fraction` is the short-range value and
`range_separated_coefficients` is `(omega, long_range_fraction, short_range_fraction)`.
Both added result metadata fields remain `None` for ordinary HF/DFT methods.

The existing fixed-occupation orbital solver accepts the same two numerical
inputs. They are dynamic leaves in its `args` dictionary, including across
`attach_scf_backward`; no integral or dispersion array is stored in static
configuration. Both implicit and explicit/unrolled modes use the same energy.
The scalar orbital objective evaluates that same local point energy directly;
it does not construct an unused Vxc/Fock derivative graph. Final state/Fock
evaluation retains the potential path. This private branch reuses the same
J/K assembly and shared energy algebra, with no change to solver tolerances,
iteration policy, or training/cache strategy.
Closed-shell RKS uses `minimize_roks_from_integrals` with equal alpha/beta
occupations. UKS uses `minimize_uks_from_integrals`. Open-shell shared-orbital
ROKS and spinor GKS are outside this extension.

The dedicated `tdscf.TDA` / `tdscf.TDDFT` adapter supports closed-shell RKS
singlets with full AO ERIs. It reuses the converged reference and shared
response/eigen solvers. The Hartree action uses the full Coulomb tensor and
the exchange action uses both tensors with the coefficients above. The local
singlet kernel projects the same SCF spin Hessian onto equal alpha/beta
perturbations, retaining the SCF density cutoff. In `RestrictedMolecule`,
`eri_lr` remains a dynamic leaf; the fixed coefficients and `xc_density_floor`
are static metadata. TDA and full TDDFT excitation energies and oscillator strengths are
checked against PySCF, including complete A/B matrices after explicitly
aligning the independent SCF orbital phases. CHG is geometry-only and does
not add a shift to vertical excitation energies or a density-response kernel.
The LR-tensor derivative checks hold the SCF reference orbitals, energies and
grid fixed. They validate explicit response inputs, not a relaxed SCF
geometry/parameter derivative of the excited state. Raw `RestrictedMolecule`
inputs retain the caller's responsibility for consistency; facade conversion
keeps the existing converged-state and freshness checks.

This TD extension does not implement triplet or unrestricted RSH response,
DF/RIS exchange, or custom XC overrides. The general single-exchange
reference boundary stays closed: RSH `to_reference` / `as_reference` and the
GW state-signature adapter still reject these states. GW, DFPT and HF-only
CC/CI guards remain in force. The restricted TD adapter is the only exception
and validates both ERI channels before constructing its response action.

Focused checks include `tests/test_wb97xd_local.py`,
`tests/test_wb97xd_stable_local.py`, `tests/test_wb97xd_scf.py`,
`tests/test_wb97xd_geometry.py`, and
`tests/test_wb97xd_tddft.py`, plus
`tests/integrals/test_native_long_range_eri.py`. The SCF reference compares
PySCF's electronic `wb97xd` result plus a separately evaluated original CHG
formula; bare PySCF `xc='wb97xd'` does not add that dispersion correction.
