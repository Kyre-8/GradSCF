# Private native integral backend

This directory builds a private CPU library with JAX typed FFI entry
points. Runtime evaluation does not import or load PySCF. The source subset is
pinned to PySCF v2.13.0 and libcint v6.1.3; full commit IDs, original relative
paths, Git blob IDs and SHA256 checksums are in `vendor/manifest.json`.
Upstream Apache-2.0 licenses are retained at `vendor/pyscf/LICENSE` and
`vendor/libcint/LICENSE`. No vendor patches are currently needed.

## Directory layout

- `__init__.py`, `build.py`: private library loading and offline build CLI.
- `csrc/ffi.cc`: GradSCF C++ value FFI adapter.
- `csrc/geometry.cc`: streaming analytic coordinate JVP/VJP drivers.
- `csrc/geometry_hessian.cc`: analytic bilinear coordinate Hessian products and adjoints.
- `csrc/coefficients.cc`: shell-local coefficient JVP/VJP and coefficient Hessian products.
- `csrc/exponents.cc`: analytic Gaussian-exponent JVP/VJP through Cartesian angular raising.
- `csrc/tables.h`: shared table validation and owned libcint buffers.
- `csrc/parallel.h`: bounded shell workers and parameter-gradient reduction.
- `vendor/`: pinned, unmodified upstream C sources, licenses and checksums.
- `include/`, `exports.*`, `CMakeLists.txt`: build configuration and ABI exports.
- `patches/`: documentation of upstream adaptations.
- `build/`, `lib/`: generated build products, excluded from Git and packaging.

Source distributions and wheels include the
native sources; compiled libraries are built locally for the installed JAX.

## Offline source build

From the repository root, with JAX, CMake >= 3.20 and a C99/C++17 compiler already
installed:

```sh
PYTHONPATH=src python -m gradscf.integrals._native.build
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q tests/integrals/test_native_values.py
```

The build verifies every vendored file before invoking CMake. It downloads
nothing and writes generated headers/object files under `src/gradscf/integrals/_native/build/`; the
private library goes into `src/gradscf/integrals/_native/lib/`. Both output directories
are ignored by Git. An installed editable package does not need `PYTHONPATH`.
Rebuild against the installed JAX headers after changing JAX versions.
Linux and macOS are supported build targets; no binary wheels are provided by
this initial source build. `--jobs` defaults to 2 for compilation only. Integral
evaluation is serial by default and does not modify process-wide threading settings.

## Parameter VJP threading and reuse

The optimized implementation is the sole maintained native path. A thread count
of 1 uses the same workspace reuse and derivative kernels as parallel execution;
it is not a legacy backend. Rebuild this library when updating the native sources.

Set `GRADSCF_NATIVE_THREADS=4` before execution to parallelize coefficient VJP,
coefficient Hessian-VJP, and exponent VJP. The default is 1; positive integer
values are capped by the number of orbital shells. Other kernels, including
JVP, retain their existing execution paths. `OMP_NUM_THREADS` is not inherited.
Use `MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` to avoid nested BLAS workers.
The limit applies **per FFI call**, not to all concurrent JAX calls combined;
budget for callback concurrency when choosing it within a Slurm allocation.

Workers own independent mutable BAS/ENV tables and shell buffers. They reduce
small parameter vectors in fixed worker order; no primitive ERI or integral
Jacobian is cached. Changing thread count can change floating-point summation
order. Within each call, real 1e/3c products reuse libcint cache sizes by intor
and shell shape; exponent products also reuse angular-raising maps and spherical
scratch. These caches do not survive a call or retain stale basis parameters.

Run `tests/integrals/test_native_parallel.py` for serial/parallel checks. The
standalone `tools/benchmark_native_basis_vjp.py` compares thread counts in the
current library with identical CPU float64 inputs and reports warm timings
separately from the first call. It accepts no alternate/legacy library path.
Its timings exclude SCF, GPU transfers, and the neural network. For example:

```sh
PYTHONPATH=src python tools/benchmark_native_basis_vjp.py \
  --threads 1 8 16 --output artifacts/native-vjp-timings.json
```

All vendor symbols have hidden visibility and a linker export allowlist;
The derivative exports include `GradSCFGeometryJVP`, `GradSCFGeometryVJP`,
`GradSCFGeometryHessianJVP` and `GradSCFGeometryHessianVJP`, plus
`GradSCFCoefficientJVP`, `GradSCFCoefficientVJP`,
`GradSCFCoefficientHessianJVP` and `GradSCFCoefficientHessianVJP`, and
`GradSCFExponentJVP`/`GradSCFExponentVJP`; the export lists
also contain the value, ECP, packed-ERI and J/K entry points. There is no dynamically
linked libcint, PySCF, OpenMP, or Python library. The build links an existing
platform BLAS provider (Accelerate on macOS). Loading uses `RTLD_LOCAL`.

## Contract

```python
from gradscf.integrals.backends.native import evaluate
values = evaluate("overlap", atm, bas, env, nao, cart=True)
```

`atm` is a static NumPy int32 `(natm, 6)` array, and `bas` is a static NumPy
int32 `(nshell, 8)` array, using libcint's public table layout. The dynamic
`env` is a JAX float64 vector. The caller supplies normalized contraction
coefficients in libcint convention, coordinates in Bohr, and zero
`env[PTR_RANGE_OMEGA]`. Enable JAX x64 before constructing arrays. Tables may
be closed over by `jax.jit`; changing coordinates/exponents/coefficients in
`env` updates values without replacing the static topology.

| Operator | Shape | Meaning / units |
| --- | --- | --- |
| `overlap` | `(nao, nao)` | S, dimensionless for normalized AOs |
| `kinetic` | `(nao, nao)` | -1/2 Laplacian, Hartree |
| `nuclear` | `(nao, nao)` | Electron–point-nucleus attraction, Hartree |
| `dipole` | `(3, nao, nao)` | Position r relative to `env[1:4]`, Bohr; no electron charge factor |
| `eri` | `(nao, nao, nao, nao)` | Chemists' `(ij|kl)`, Hartree, full s1 tensor |

Both Cartesian and real spherical AOs use libcint/PySCF ordering and
normalization. This initial interface supports point nuclei, scalar Gaussian
orbitals with angular momentum 0..12, and ordinary Coulomb integrals. It
rejects nonfinite environments, nonpositive exponents, unsupported nuclear
models/range separation, invalid pointers and inconsistent AO counts.
ERI shell workspaces that exceed the upstream int32 indexing range are rejected
before allocation. The adapter owns the ERI workspace with checked C++ allocation
and invokes the unmodified PySCF `GTOnr2e_fill_s1` filling kernel.
This check includes a Cartesian contraction-cache lower bound even for spherical
output. Every native shell cache query must also return a strictly positive size;
libcint's zero-on-overflow sentinel is returned to JAX as an error before filling.

## Coordinate derivatives

`integrals.make_plan(..., backend="native").evaluate(...)` supports first- and second-order
`jax.jvp`, `jax.vjp`, `jax.grad`, `jax.jacfwd` and `jax.jacrev` for nuclear
coordinates and basis centers, including JIT/batching. Dipole origins also
participate in AD; a default charge-center origin follows nuclear motion.
C++ evaluates analytic derivative shell blocks and immediately contracts them
with the tangent or cotangent. No full coordinate Jacobian or Python callback
is used. Nuclear-attraction operator motion and AO-center motion are separate.

For an integral tensor `I(R)`, the second-order kernels evaluate
`H[u,v] = sum_ab (d²I/dR_a dR_b) u_a v_b` and its coordinate adjoint
`sum_bI (d²I/dR_a dR_b) u_b cot_I`. Both contract analytic libcint shell
blocks directly. Nested JVPs, forward-over-reverse HVPs, reverse-over-reverse
Hessians and derivatives with respect to tangent/cotangent vectors share these
two kernels. Independent nuclear/basis coordinates and dipole origins include
all mixed terms. No finite differences are used by the implementation.

```python
from dataclasses import replace
import jax
import jax.numpy as jnp
from gradscf import integrals

jax.config.update("jax_enable_x64", True)
topology, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 .74", basis="3-21g")
plan = integrals.make_plan(topology, backend="native")

def overlap_sum(centers):
    return plan.evaluate("overlap", replace(parameters, centers=centers)).sum()

centers = parameters.centers  # Bohr; move the basis centers independently here.
direction = jnp.zeros_like(centers).at[0, 2].set(1.)
hvp = jax.jit(lambda r, v: jax.jvp(jax.grad(overlap_sum), (r,), (v,))[1])
product = hvp(centers, direction)
hessian = jax.jit(jax.hessian(overlap_sum))(centers)
```

An SCF energy Hessian additionally needs the converged orbital/density response
and nuclear repulsion derivatives. Integral Hessians alone, or differentiating
a frozen-density force expression, do not supply that response.

The low-level raw-ENV `backends.native.evaluate` remains value-only: an arbitrary
ENV vector mixes geometry, exponents and coefficients. Use integral plans for
geometry and basis-parameter AD. Third-or-higher coordinate derivatives remain
unsupported and fail explicitly. Geometry AD is limited to the five full
operators listed above: ECP, RI, packed ERIs and fused J/K do not acquire
geometry derivatives through this change. GPU FFI, periodic integrals, spinors
and range separation are outside this interface.
Full integral values still require O(nao^4) memory.

## Contraction-coefficient derivatives

For `overlap`, `kinetic`, `nuclear` and `dipole`, integral plans accept raw
coefficient JVP/VJP, forward/reverse Hessians, HVP and batching. JAX retains the
primitive and contraction normalization in `normalization.py`. The native
drivers differentiate the normalized coefficients stored in libcint ENV;
normalization is not duplicated in C++.

Each orbital leg uses an independent shadow shell, including when both legs
refer to the same original shell. JVP replaces one leg with its coefficient
direction. VJP locally uncontracts only the differentiated leg and immediately
reduces the integral block against the cotangent. Zero and negative individual
coefficients are supported; no primal nonzero-coefficient optimizer mask is
reused. No complete coefficient Jacobian or primitive-pair integral cache is
created. ENV coefficient slots must be disjoint from other coefficients,
exponents and coordinates. Cartesian workspace checks apply to spherical
output as well, including the larger identity contraction used by VJP.

At fixed geometry/exponents these integral kernels are quadratic in the
normalized orbital coefficients. Native second products therefore do not
depend on those coefficients; their coefficient third derivative is exactly
zero. Raw coefficient derivatives of higher order remain nonzero through the
JAX normalization chain. Coefficient/geometry mixed second derivatives are
currently rejected. Dense simultaneous coefficient/coordinate/origin
**first-order** chain rules are supported.

```python
from dataclasses import replace
import jax
from gradscf import integrals

top, params = integrals.prepare_basis("H 0 0 0; H 0 0 .74", "3-21g", cart=False)
plan = integrals.make_plan(top)
kinetic = lambda coefficients: plan.evaluate(
    "kinetic", replace(params, coefficients=coefficients))
gradient = jax.jit(jax.grad(lambda c: kinetic(c).sum()))(params.coefficients)
```

Packed RI three-center orbital-coefficient AD requires a **precomputed fixed
auxiliary metric**. Cache signature and `lindep` checks stay active. Whitening
uses JAX triangular solves or a fixed eigenmode whitening matrix, so the
coefficient cotangent returns through the native three-center VJP.

```python
aux_top, aux_params = integrals.prepare_basis(
    "H 0 0 0; H 0 0 .74", "def2-universal-jkfit", cart=False)
ri = integrals.make_auxiliary_plan(top, aux_top)
metric = ri.metric_factor(params, aux_params)  # outside JIT/AD
factors = lambda c: ri.factors(
    replace(params, coefficients=c), aux_params, metric_factor=metric)
gradient = jax.jit(jax.grad(lambda c: (factors(c)**2).sum()))(params.coefficients)
```

The cached RI path supports orbital coefficient JVP/VJP/HVP at fixed auxiliary
basis and geometry. Auxiliary-parameter and RI geometry AD remain
explicitly unsupported; a cached host metric cannot stand in for their
derivatives. Packed four-center ERIs, ECP and shell-direct J/K do not acquire
coefficient AD through this change. Native FFI remains CPU/float64; GPU SCF
requires an explicit transfer of contracted physical inputs/cotangents.

The coefficient tests compare native products with fourth-order finite
differences and an independent JAX primitive contraction, using s/p/d general
contractions in Cartesian/spherical representations. End-to-end H2 checks
compose contracted native S/H/RI directly with implicit SCF and compare its
energy, coefficient gradient and HVP with the fixed-primitive reference. These
checks store no complete four-center ERI and do not establish a molecular force
Hessian or force-loss support for the new coefficient/geometry combination.

## Gaussian-exponent derivatives

Plans support first-order orbital exponent JVP/VJP for overlap, kinetic,
nuclear attraction and dipole integrals, and for packed three-center RI with
a fixed auxiliary metric. Exponents are shared across all contraction columns
of a shell. JAX differentiates primitive/contraction normalization; C++ supplies
the derivative of the bare Gaussian, `d g / d alpha = -r^2 g`. Both terms are
required, including for a normalized single primitive.

The native driver raises the differentiated Cartesian leg by two powers along
each axis, sums the three terms, and then transforms the **original** angular
shell to spherical AOs. The ratio of libcint's s/p common factors compensates
for the changed shell convention. Each primitive uses an independent shadow
shell and each VJP block is reduced immediately, without a full exponent
Jacobian, primitive ERI cache or four-center ERI tensor. Exponent AD requires
orbital `l <= 10` because the raised shell must fit the compiled `l <= 12` limit.

```python
def kinetic_from_log_scales(beta):
    alpha = tuple(a * jax.numpy.exp(b) for a, b in zip(params.exponents, beta))
    return plan.evaluate("kinetic", replace(params, exponents=alpha)).sum()

beta = tuple(jax.numpy.zeros_like(a) for a in params.exponents)
gradient = jax.jit(jax.grad(kinetic_from_log_scales))(beta)
```

Joint first-order exponent/coefficient differentiation is supported. Exponent
Hessians and mixed exponent/coefficient/geometry derivatives fail explicitly;
pure coefficient HVP support is unchanged. ECP, four-center and direct J/K
exponent AD, auxiliary-basis AD and GPU FFI remain unsupported. Tests compare
Cartesian/spherical s-through-g products against finite differences and verify
the complete log-exponent/coefficient gradient through reconverged implicit
DF-RHF. A small H2 joint Adam test establishes variational descent, not shared
MACE training or basis-set accuracy across molecules.

## Why these upstream files

PySCF's `pyscf/lib/gto/fill_int2e.c` provides the shell-to-full-tensor ERI
driver. Its only headers are libcint's `cint.h` and a local build `config.h`.
One-electron assembly lives in the GradSCF adapter and invokes libcint shell
kernels directly, avoiding PySCF's unrelated np_helper/BLAS dependencies.
Shared libcint optimizer objects also reference grid and two-/three-center
environment helpers; those small dependencies are included even though no
such operator is exposed. Large optional polynomial-fit tables, F12 code,
other generated operators, and upstream build/test machinery are excluded.
The pinned `autocode/grad1.c` and `autocode/grad2.c` provide the one-/two-electron
first-derivative kernels; their blob and SHA256 hashes are in the manifest.
The pinned `autocode/hess.c` supplies the second-derivative kernels.
All vendored files remain byte-identical to their pinned upstream blobs.

The numerical tests compare identical tables with PySCF at absolute tolerance
3e-11 and relative tolerance 3e-12, covering water/6-31g*, Cartesian and
spherical forms, eager/JIT values, and dynamic coordinate changes. A separate
process blocks all PySCF imports and checks a primitive overlap analytically.
Tests also verify unsupported AD and malformed-input rejection.

Second-coordinate tests compare analytic products with fourth-order finite
differences of first derivatives (CPU float64, step `2e-4` Bohr), including
independent nuclei, floating centers, origin motion and d shells. They also
check nonsymmetric cotangents, Hessian symmetry, JIT/batching and PySCF-free
execution. For H2/3-21G at 0.74 Angstrom, differentiating the converged implicit
SCF energy twice gives `0.400305894977` Hartree/Bohr² for the second H atom's
z-coordinate curvature, versus `0.400305897150` from PySCF's analytic RHF
Hessian (absolute difference `2.18e-9` Hartree/Bohr²). The focused SCF test
independently checks this curvature against finite differences of the
self-consistent energy gradient.
