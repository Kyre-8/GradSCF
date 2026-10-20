# GaAs reference sampling and paused external checks

This is the retained result state at the local integration on 2026-10-19.
Further periodic GW/BSE development and full external scGW are deferred.
All numerical work ran on jy_gpu_1/c20, Intel Xeon Platinum 8457C,
with CUDA devices hidden. GPU6 is prohibited. Disjoint allocations remained
below the project-wide 96 logical CPUs; each job was individually bounded.

## Coverage

| Method | Independent implementation | Accepted result / scope |
| --- | --- | --- |
| G0W0 | Original BerkeleyGW 4.0, EPM/GPP | Gamma and sampled gaps 3.424335267 eV on Gamma-containing 4^3 and 8^3 queries |
| evGW | Original BerkeleyGW 4.0, EPM/GPP; G and W updated | Not converged: all-18-state physical residual 0.012018824321 eV versus 1e-5 eV; no accepted gap |
| qsGW | Original ecalj, PBE all-electron PMT, mode A | 12 outer iterations; Gamma and sampled gaps 2.06610 eV on 4^3 and 8^3 queries |
| scGW | Original GREEN, exact native finite Hamiltonian | One unmixed GW map agrees; full external self-consistency was not submitted |

BerkeleyGW EPM/GPP and ecalj all-electron PMT differ from the native
GTH-SZV/CD finite model. Their absolute gaps are not matched accuracy errors.
See [evGW status](evgw_summary.json) and [ecalj convergence/sampling](qsgw_sampling.json).
The independently completed G0W0 baseline appears inside the evGW summary;
failed evGW states are not used for its fine queries.

## Exact-Hamiltonian GREEN comparison

Both programs use the saved native GaAs source model: a=5.6533 Angstrom,
PBE/GTH-SZV/GTH-PBE, FFT61^3, source mesh 2x1x1, 14 Cartesian orbitals,
18 electrons, separate bare/full-FFT and correlation/16-Ry factors,
no SOC and body-only Coulomb. The fixed source manifest is
[source_model_manifest.json](source_model_manifest.json).

The SRC input adapter changes auxiliary storage to GREEN's pair-adjoint
convention while certifying the full cross-k Coulomb Gram. It does not add
a different DF approximation. Native and external calculations use the same
initial Fock and chemical potential 0.29783017174122506 Ha, beta=400 Ha^-1.
Native nw1024/Gauss8192 and the official GREEN IR grid share 92 exact
Matsubara nodes. No frequency interpolation or fitted shift is used.

| Full matrix check | Maximum absolute difference |
| --- | ---: |
| Correlation Sigma(iw), 92 common nodes | 7.17673e-12 Ha |
| Static J + negative exchange | 8.45305e-12 Ha |
| One-spin density | 6.44643e-11 |
| Self-energy endpoint moment | 2.07882e-11 Ha^2 |

The [full comparison](green_one_map_comparison.json) records frequency
indices, input/output hashes, units, density convention and frequency bands.
Relative Frobenius error in correlation Sigma is 1.54086e-11.
The [native map record](green_native_map_resource.json) gives 184.976 s
monitored wall time and 36.19 GiB peak process-tree RSS, CPU64--79.
The external map ran in 24.795 s on one rank or 3.497 s on 16 ranks,
CPU32--47. These are single measured runs, not warm performance benchmarks.

The original GREEN HF 1e-11 Hermiticity gate failed and is retained in
[its raw report](green_hf_failed_gate.json). The separate
[endpoint diagnostic](green_hf_endpoint_diagnostic.json) identifies an
approximately 1e-10 IR endpoint error. Inputs/checkpoints were not changed
to make that stricter gate pass. The
[MPI comparison](green_mpi_comparison.json) records identical dynamic Sigma
and static differences below 1.8e-15 Ha. Original source pins are preserved
in [green_source_pins.json](green_source_pins.json). Old export metadata uses
the semantic spelling ignore_G0; executed CLI uses the required IGNORE_G0.

These results validate one physical GW update and its input bridge.
They do not certify a full scGW fixed point, continuation or band gap.

## Native source versus queried band edges

[Four-method JSON](four_methods.json) and [CSV](four_methods.csv) retain the
existing native gap and L-star diagnostics. The evGW Gamma/sample gaps are
1.033671314/0.606994808 eV and qsGW gives 1.299711663/1.200526646 eV.
The higher queried VBM lies at (-0.5,-0.5,-0.5), absent from the 2x1x1
source nodes. This source mesh is not cubic-symmetry complete; even PBE
has L-star splitting. This locates a source-to-query issue without uniquely
assigning it to interpolation. Denser queries do not refine the GW source.
The native scGW gap remains a selected finite-eta ReSigma root diagnostic,
not a principal-peak or absorption-onset claim.

Large checkpoints, external source/build trees and orchestration stay
outside this selected result bundle. Their remote root is
jy_gpu_1:/home/yjiao/gradscf_checks/gw-reference/. No executable code is
stored here; physical algorithms and input adapters are in src.

## Local-integration validation

The final tested source passes 162 focused remote tests: 68 shared-solver /
public-interface tests, 71 periodic-kernel/spectrum tests and 23 example /
reference-adapter tests. All three processes exit zero with no failed,
errored or skipped tests. This is a focused regression, not the full suite.
The [validation records](merge_validation/) retain exact commands, JUnit,
stdout, CPU limits, wall times and source/test/example hashes. No local
numerical tests or long GW calculation were run for this integration.
