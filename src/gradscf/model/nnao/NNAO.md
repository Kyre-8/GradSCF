# MACE-conditioned SZP3 basis assembly

This module connects an upstream MACE-JAX encoder to per-atom contracted
Gaussian basis parameters. The methane tool validates training against converged RHF energies.
The upstream snapshot remains unchanged; `vendor/mace-jax/UPSTREAM.json` identifies its commit
and checksums. Our adapter files are `basis.py`, `graph.py`, and `encoder.py`.


## Current primary path: all-electron 4s4p2d

`prepare_direct_basis` and `MACEBasisModel` now default to `szp442_direct`.
P-block main-group elements use four s primitives, four p primitives and two
d primitives contracted into one radial function per angular channel. Inner
shells default to six fixed primitives from bundled 6-31G for Li–Ca.
H/He have no inner shells. The s-block valence retains 3s + fixed 1p;
no new angular shells are added. Heavier elements need explicit
`core_primitives=3` (the original 3-21G cores); six-primitive heavy-element
parameters are not yet validated, and requesting them raises an error.
There is no ECP or core-electron subtraction in this path.

The new pool preserves every preceding primitive. Each expanded s/p/d shell
adds exponent `min(old_exponents)/2` with initial coefficient zero. This is a
deterministic diffuse extension, not a published or variationally fitted set
of exponents. With `core_primitives=3`, zero-head initialization reproduces the preceding
three-primitive-valence basis exactly. The new six-primitive core replaces
both core exponents and coefficients, so it changes the initial basis.
The old template data are not mutated. All exponents remain fixed in training.

The output shape is `(natom, 3, 4)` (s/p/d, padded primitive coefficients).
MACE predicts full signed raw vectors with no additive empirical baseline.
For p-block atoms the active lengths are 4/4/2; for H the active s length is 3.
Core shells and single-primitive polarization shells remain fixed; the two-
primitive d contraction is trainable. Each shell is normalized separately.
A zero/nonfinite active shell is invalid; padded entries cannot affect a shell.

The AO count is unchanged: methane has 10 electrons and 26 spherical AOs
(27 Cartesian). With six-primitive cores there are 56 primitive spherical AOs (58 Cartesian),
compared with 53 (55) for core-3 + 4s4p2d. Full primitive ERI caching therefore grows despite
unchanged SCF matrix dimensions. It remains a bounded validation implementation.

The methane CLI defaults to `--basis-family szp442_direct` with six-primitive
cores. Use `--core-primitives 3` for the preceding core-3 + 4s4p2d run. Select
`szp3_direct` explicitly for the preceding direct three-primitive comparison;
`szp3` and `qvszps` are explicit tangent/ECP comparison modes. This expanded
all-electron design is not the published Grimme q-vSZPs basis.

## Original SZP3 scope and initialization

`szp3-initial` includes 34 main-group elements through Xe: H/He, Li–Ne,
Na–Ar, K/Ca/Ga–Kr, and Rb/Sr/In–Xe. Transition metals are not NNAO templates.
GradSCF molecule parsing and molecular-grid constants separately cover Z=1–54.
Data coverage does not imply validated heavy-element chemistry, ECP, or
relativistic accuracy. Periodic GTH coverage is an independent contract.

Core shells retain the bundled 3-21G exponents and coefficients. Fixed core
basis parameters do not freeze core electrons in a future SCF calculation.
Each trainable outer s/p radial function uses the union of the 3-21G split
valence 2+1 primitives. Reference coefficients are an equal-amplitude sum of
the individually normalized split contractions. This is a reproducible
initialization, not an optimized basis or a reproduction of q-vSZPs.

H/He and alkali/alkaline-earth elements train only the outer s shell and have
one fixed single-primitive p polarization shell. Other main-group elements
train independent outer s and p contractions and have one fixed d polarization
shell. Polarization exponents come from the last single-primitive shell of
the corresponding angular momentum in bundled def2-SVP, as documented in
`data/szp3.json`. No def2 ECP is imported or applied. Br/I core d shells remain
explicit, fixed core shells and are distinct from the added d polarization.

Both Cartesian and real spherical integral layouts are supported. High-level
GradSCF SCF/grid assembly still requires Cartesian AOs; choosing `cart=False`
here does not remove that higher-level restriction. Gaussian normalization
follows GradSCF/libcint conventions, including Cartesian angular factors.

## Basis API (no MACE dependency)

```python
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from gradscf.model.nnao import prepare_basis
from gradscf import integrals

layout = prepare_basis('H 0 0 0; I 0 0 1.6', unit='Angstrom')
outputs = jnp.zeros((2, 2, 2))  # atom, valence s/p slot, tangent coordinate
parameters = layout.bind(outputs)
overlap = integrals.make_plan(layout.topology).evaluate('overlap', parameters)
```

Zero outputs recover the reference basis. Every three-primitive contraction
uses two coordinates in the Euclidean tangent plane of its reference vector;
`tanh` bounds updates, and the reference norm fixes the common-scale gauge.
Signs are allowed. The integral layer performs physical Gaussian normalization.
Inactive slots never alter frozen core or polarization shells.
`layout.bind(outputs, coords_bohr=...)` accepts differentiable coordinates.
`layout.atom_shells(parameters)` exports separate raw shell lists for each atom;
it is host-only and never merges chemically distinct atoms of one element.

## MACE API and optional dependencies

Install the optional stack in a dedicated environment:

```sh
python -m pip install -e '.[nnao]'
```

The extra pins MACE-JAX to the upstream commit. Upstream requires JAX >=0.10,
Flax NNX, cuEquivariance, and e3nn/PyTorch for CG constants even though neural
execution and differentiation use JAX. No dependency installation is performed
by importing GradSCF or by the adapter. Developers may instead install the
unchanged snapshot as `pip install -e src/gradscf/model/nnao`.

```python
from flax import nnx
from gradscf.model.nnao import MACEBasisModel, build_graph

model = MACEBasisModel(elements=(1, 53), rngs=nnx.Rngs(0))
graph = build_graph([1, 53], [[0, 0, 0], [0, 0, 1.6]],
                    element_order=model.elements, total_charge=0, spin=0)
parameters = model.assemble(layout, graph)
```

The model uses only the final invariant `0e` features, then element-specific
heads. Original MACE energy outputs are not a surrogate for SCF energy.
Charge/spin are graph-level conditioning inputs, not predicted atomic charges.
Zero initialization of the final head reproduces the reference basis exactly;
encoder gradients start after that head acquires nonzero weights.

Graph distances/cutoffs are Angstrom; integral coordinates are Bohr and exponents
Bohr^-2. `build_graph` is a host operation with quadratic neighbor search.
Sorted contiguous `batch` indices support disconnected molecular graphs.
There are no padding atoms or periodic graphs in this initial adapter.
`graph.with_positions(...)` preserves coordinate differentiation on fixed edges;
rebuild the neighbor graph outside AD when atoms move beyond the neighbor skin.
Element order, atom order, and cutoff must agree with the model/layout.

## Coefficient derivatives and validation

`gradscf.integrals.basis.contraction` exposes `primitive_basis`,
`contraction_matrix`, and `contract_integrals`. With fixed exponents and
coordinates, compute native primitive tensors once, then differentiate JAX
contraction/normalization with respect to network outputs or model weights.
Native `plan.evaluate` supports coefficient JVP/VJP and second products for
overlap, kinetic, nuclear-attraction and dipole integrals. Fixed-auxiliary
three-center RI supports coefficient products when a metric is prepared before
AD; these paths also support first-order orbital exponent derivatives.
`tools/train_nnao_joint_ad.py` uses contracted native integrals and strict
implicit SCF to optimize MACE coefficients and shared log-exponent scales.
Its DF-RHF primal checks real internal orbital stability and follows negative
curvature with bounded SCF restarts before attaching the implicit response.
Unresolved or unstable candidates are rejected without advancing parameters
or Adam moments. Small orbital spaces use a Hessian; larger ones use the
shared HVP/Davidson solver. This does not test spin/complex instabilities or
certify a global minimum. Density-only guesses use a density-consistent J/K.
The initial checkpoint parity check retains the pre-stabilization SCF energy;
`parent-parity.json` records stable energies separately as the new loss baseline.
Older checkpoints must be revalidated under this policy before resuming training.
The joint trainer bounds resident CPU integral JIT kernels with
`--cpu-jit-cache-size` (default 8), retaining GPU model/SCF kernels when possible.
Linux executable-map pressure triggers additional eviction before dispatch.
Disk compilation caching is enabled under the output directory by default;
`--compilation-cache-dir` selects another root, always namespaced by the native
library hash. Cache eviction does not modify basis parameters or optimizer state.
Four-center ERI, ECP and direct-J/K basis derivatives, exponent Hessians and
basis/geometry mixed second derivatives are not provided by this path.
A full primitive ERI cache costs O(N_primitive^4) and is confined to bounded
reference checks; the joint native trainer creates no such cache.

Tests cover all template layouts, per-atom/per-shell mapping, native versus
PySCF integrals for HF/HCl/HBr/HI, Cartesian/spherical representations, JIT,
coefficient finite differences, MACE rotation/permutation invariance, zero-head
initialization and model-weight finite differences. Real MACE tests require the
optional stack; skipped tests are not evidence of model validation.

## Direct Grimme-family coefficients and scalar ECP

Use `prepare_grimme_basis(atom, cart=False)` and
`MACEBasisModel(..., basis_family='qvszps')` for the new direct-output path.
This uses the official qavg-vSZPs primitive pool and matching ECP, pinned with
license/provenance in `data/qvszps/`. H/He use 5s1p primitives; C/N/O/F use
4s4p2d. Each angular channel contracts into one radial function. Other supported
main-group templates retain the official per-element counts.

The neural tensor has shape `(natom, 3, 5)`: s/p/d channels, up to five primitive
coefficients each. Padding is masked. The trainable output bias is initialized
from the normalized official averaged coefficients. Every forward value is
`W(features) + bias`; the assembler consumes it directly and never adds a fixed
reference vector or tangent correction. Coefficient signs are unrestricted.
Common scale is removed, then existing Gaussian normalization is applied.
Zero/nonfinite contractions fail eagerly and yield NaNs under tracing.

`layout.topology.nuclear_charges` contains effective charges Z-ncore;
`layout.ecps` holds explicit scalar ECP terms; `layout.nelectron` subtracts
removed core electrons and the molecular charge. `layout.symbols` and MACE
features retain the physical elements. This differs from the all-electron SZP3
layout. For methane the new spherical basis has 25 AOs and 8 explicit electrons.

The one-electron Hamiltonian is `kinetic + nuclear + ecp`, where the nuclear
operator and nuclear repulsion use effective core charges. Evaluate ECP as
`plan.evaluate('ecp', parameters, ecps=layout.ecps)`. The backend compiles pinned
PySCF scalar ECP C sources into GradSCF's private library; no PySCF Python call
or libcgto library is used. Rebuild the native library after this update.
macOS links system Accelerate; Linux requires a BLAS library at build time.

Scalar ECP values support JIT, orbital l<=3, ECP channels <=4 and radial powers
0..6. Spin-orbit ECP and direct ECP geometry/exponent AD are not implemented.
Coefficient AD is supported by native fixed primitive ECP values followed by
JAX contraction. Do not claim force-training support for this new ECP path.
High-level molecular SCF/grid facades do not yet consume ECP metadata; use the
explicit integral interface, as in the methane driver, which adds the ECP term.

Run the controlled methane experiment with
`tools/optimize_methane_nnao.py --basis-family qvszps`. Compare its starting
qavg-vSZPs energy and learned coefficients under the same ECP Hamiltonian;
absolute totals must not be compared with the previous all-electron energies.

## Expanded primitive experiment: szp663_direct

Select `basis_family="szp663_direct"` for p-block 6s6p3d and H/He/s-block
6s2p, with the same contracted angular shells as szp442_direct. Each shell
keeps its old exponents in their original order. Extra exponents are twice
the old maximum, then half the old minimum, then the geometric midpoint of
the widest logarithmic gap if needed. These are deterministic experimental
exponents, not published basis parameters. New coefficients initialize to zero.
The H/s-block two-primitive p contraction becomes trainable; core parameters
remain fixed. The head has shape `(natom,3,6)`. CH4 retains 26 spherical AOs
but has 93 primitive spherical AOs, versus 56 in core6 + szp442_direct.

The methane CLI accepts `--initial-summary <previous-summary.json>` to embed
saved contractions in the expanded pool. It initializes the trainable head
bias, with a seed-0 backbone and zero head kernel, and verifies that the first
energy reproduces the saved energy. This transfers contraction coefficients,
not a whole network checkpoint. No fixed reference vector is added in forward.
The preceding szp442_direct default remains available for reproducibility.
