# Pinned MolGW reference environment on jy_gpu_1

Built on 2026-10-16 (Asia/Shanghai) in the isolated directory
`/home/yjiao/gradscf_checks/ad-showcase-molgw-reference-20261016`.
The stock build passed the three existing H2/STO-3G G0W0, evGW0 and evGW
comparisons, including nine singlet/triplet TDA/full-BSE comparisons, using
the original acceptance tolerances. This is a molecular reference component,
not validation of nuclear GW/BSE response or every MolGW feature.

## Source and dependency boundary

MolGW 3.4 is pinned to
[`b831818d7a845c36f295d036dc8ceef59daf9991`](https://github.com/molgw/molgw/tree/b831818d7a845c36f295d036dc8ceef59daf9991).
Every one of the 3529 files in the 5,601,560-byte official archive was checked
against its fixed-commit Git blob SHA1. The archive SHA256 is
`dcf0c147f3dda2b7fdbedcb126c8d5063e0d869e6a1bdbc05d4d4cd8cf72bd6d`.
See [source verification](source_verification.json) and the
[complete blob manifest](molgw-source-blob-sha1.json).

The system GCC/GFortran is 8.5.0. The build instead uses the already installed
Intel Fortran 2021.9.0 from oneAPI 2023.1 and system G++ 8.5.0; CMake is 3.20.2.
No compiler or shared Python environment was installed or upgraded.
The stock MolGW Fortran flags are `-fpp -O2`, with no MolGW OpenMP build.
Existing PySCF libraries supply OpenBLAS LP64 and LibXC 7.0.0. Their location
is `/home/yjiao/opt/miniconda3/envs/jax_scf/lib/python3.11/site-packages/pyscf/lib`.
The OpenBLAS file is `libopenblas-r0-f650aae0.3.3.so`; that directory must be
on `LD_LIBRARY_PATH` for its bundled GFortran/quadmath dependencies.
Its runtime configuration reports `SINGLE_THREADED` and one actual thread;
the environment setting of two threads is not an observed two-core allocation.

The existing PySCF libcint 6.1.3 lacks the old CINT2 symbols used by this
MolGW version. A **separate** libcint 6.1.3 build therefore uses official
commit `c72d30662ba08048940c9000cc352a220c867712`, with
`WITH_CINT2_INTERFACE=ON`, `WITH_FORTRAN=ON`, `WITH_RANGE_COULOMB=ON`,
`WITH_F12=OFF` and `WITH_POLYNOMIAL_FIT=OFF`. Its 124 unmodified source files
total 3,570,990 bytes. Nine large tables used only by the disabled optional
F12/polyfit branches are omitted; the ordinary `roots_for_x0.dat` table is
retained. Every included file has its upstream Git blob verified. The
[source manifest](libcint-source-verification.json) records both included
files and omitted tables. No GradSCF source, native integral library, PySCF
library or system library was replaced.

The verified small libcint archive is 713,781 bytes, SHA256
`9d03e83d368cf045b39da772a278a13fb265dd9fe1c64d0c91740054a12d4c3d`.
The private installed libcint binary SHA256 is
`d6d06ccb2f9196f1c180f5887d3e95f21641110a20324947a3d5752a58a3f6df`.
The first subset attempt omitted the ordinary table too; its compile failure
is retained in [the log](../../../artifacts/ad_showcase/provenance/MolGW_reference_build/libcint-missing-ordinary-table.log). The correction
restored the original verified file, without modifying numerical kernels.

## Three separate reference executables

| Executable, relative to the remote directory | Source boundary | SHA256 |
| --- | --- | --- |
| `molgw-stock/molgw` | Unmodified pinned MolGW numerical source | `a3cb4799eaa416885b2969d5e2b219ec5f9e03a9ea2b063ab685731f663c8b58` |
| `variants/adapter/molgw` | Existing optical-window/screening-energy input adapter only | `7af684eaa37639b1d144f8c044f8e962d5df2189704a0c6a074f04c2dc3a0f53` |
| `variants/full-rank/molgw` | Same adapter plus the existing auxiliary cutoff `1e-10` variant | `a91a590b5402e5ae265adecee607e158b8fba1e29e40be41acba3a1a5d899fa9` |

The two patches are the unchanged historical
[BSE input adapter](../../gw_bse/methane_aug_cc_pvdz/molgw/bse_input_driver.patch)
and [auxiliary cutoff variant](../../gw_bse/methane_aug_cc_pvdz/molgw/auxiliary_cutoff.patch).
The patched main-program source SHA256 exactly matches the historical build.
All other stock numerical object files are reused. The
[variant build manifest](variant-build.json) records every original object
hash, replacement source/object, patch, compiler argument and linker argument.
It verifies that the stock executable and every stock object remained unchanged.
Building a variant alone does not establish its numerical comparison results.

The source was obtained as a verified archive, not a Git checkout. The generated
MolGW Git banner is therefore `-1`; the external blob/archive manifests identify
the actual pinned source. The H2 driver reuses the existing comparator's
`compare_case` and original numerical gates, while replacing only the CLI's
Git-checkout introspection with the archive-verification record. It does not
fabricate a Git revision or copy any electronic-structure solver.

## H2 baseline actually executed

The existing molecular comparator uses H2 at `0.74 Angstrom`, Cartesian
STO-3G, full orbital spaces, HF SCF tolerance `1e-12`, and GW eta `1e-5 Eh`.
MolGW uses its original 40001-point, `5e-5 Eh` graphical grid for G0W0;
GradSCF uses `nw=200`. The three cases each compare TDA singlet, full singlet
and full triplet BSE; the singlet optical broadening is `0.01 Eh`.

| Quantity | Maximum observed difference | Original acceptance limit |
| --- | ---: | ---: |
| HF energy / Eh | 1.04e-10 | 1e-7 |
| QP energy / Eh | 7.79e-9 | 2e-6 |
| G0W0 QP weight | 9.58e-8 | 5e-5 |
| BSE excitation / Eh | 1.07e-8 | 3e-6 |
| Oscillator strength | 1.28e-8 | 2e-5 |
| Static polarizability / bohr^3 | 2.87e-8 | 3e-4 |
| Sampled cross section / bohr^2 | 9.78e-7 | 2e-4 |

All three cases completed in 41.476 s on the shared CPU node. The native
source is frozen `ad-showcase-integration-20261015/src`, separately from the
new GW geometry-response work. [comparison.json](h2_baseline/comparison.json)
contains all per-case/per-observable differences, source hashes and settings;
[the log](../../../artifacts/ad_showcase/provenance/MolGW_reference_build/h2_baseline/h2-baseline.log) preserves actual output. This baseline
does not independently certify outer-loop convergence for every larger system.

## Execution and retained evidence

The build and comparison used CPU float64 with `OMP_NUM_THREADS=2` and
`OPENBLAS_NUM_THREADS=2`. These are shared-node workflow observations, not
exclusive-resource performance benchmarks. Each build and H2 job was bounded
to 1800 seconds. `make -j2` bounded compilation concurrency; no large dependency
chain was installed. Build logs are [libcint](../../../artifacts/ad_showcase/provenance/MolGW_reference_build/libcint-build.log),
[stock MolGW](../../../artifacts/ad_showcase/provenance/MolGW_reference_build/molgw-stock-build.log), and [variants](../../../artifacts/ad_showcase/provenance/MolGW_reference_build/molgw-variants-build.log).

Exact recipes are in [recipes](../../../artifacts/ad_showcase/provenance/MolGW_reference_build/recipes). The verified source archives,
executables, generated modules and complete raw outputs remain in the remote
directory, with local development caches under `artifacts/ad_showcase/`.
They are not included in this small evidence record. To reproduce the build,
place the retained archives and [checksum list](archives.sha256) in the remote
`downloads/` directory, then run the recorded build scripts under the stated
timeouts. A fresh official checkout of the same commits can also be built with
the same options; the optional omitted tables are unused by this build.

The [official MolGW build documentation](https://www.molgw.org/tuto_compilation/)
describes its compiler, BLAS/LAPACK and integral-library configuration.
This record additionally fixes the actual source, options and runtime libraries
used here. Water single-point root-selection evidence is kept separately under
[water_forward](water_forward/), including rejected attempts and the distinction
between maximum-Z graphical selection and locally seeded roots.
The [methane replay record](methane_replay/) contains the separately executed
formal native resolvent example, three matched MolGW comparisons passing the
original gates, three default-RI comparisons, and the unchanged original
facade's runtime failure. The two native execution paths are not conflated.
