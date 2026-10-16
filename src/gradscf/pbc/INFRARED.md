# Gamma infrared response and the nonanalytic limit

The array functions in `pbc.vibrational` connect supplied Born charges,
electronic dielectric response and Gamma phonons. They reuse the existing
mode solver and do not solve electronic response themselves. Born charges
and the dielectric tensor can be obtained from `dfpt.electric` on a full
insulating k mesh. The force constants, charges and dielectric tensor must
describe the same cell, basis, XC, occupations and numerical discretization.

The tensor convention is

\[
Z^*_{a,ij}=\Omega\frac{\partial P_i}{\partial R_{aj}},
\qquad \epsilon^\infty_{ij}=\delta_{ij}+4\pi\frac{\partial P_i}{\partial E_j}.
\]

Charges are in units of e, coordinates in bohr, polarization in e/bohr² and
volume in bohr³. Neither Born tensors nor the electronic dielectric tensor
are assumed diagonal. `nonanalytic_force_constants` accepts a nonzero
Cartesian reciprocal-space direction and returns the **unweighted** matrix

\[
\Phi^{\rm NA}_{aj,bk}(\hat q)=\frac{4\pi}{\Omega}
\frac{(\hat q_iZ^*_{a,ij})(\hat q_lZ^*_{b,lk})}
     {\hat q_i\epsilon^\infty_{il}\hat q_l}
\]

in Eh/bohr². Add it to the short-circuit Gamma force constants before mass
weighting. Its direction dependence is the macroscopic LO–TO limit. It is
not a finite-q dipole interpolation model. See the
[Phonopy formulation](https://phonopy.github.io/phonopy/formulation.html#non-analytical-term-correction)
and [Gonze and Lee, Phys. Rev. B 55, 10355 (1997)](https://doi.org/10.1103/PhysRevB.55.10355).

For orthonormal mass-weighted Gamma modes e, `mode_effective_charges` computes

\[
z_{i\nu}=\sum_{aj} Z^*_{a,ij}\frac{e_{aj,\nu}}{\sqrt{M_a}}.
\]

The function accepts masses in amu and converts them to electron masses.
Thus a normal coordinate Q has units bohr√me, its cell dipole is zQ, and
its harmonic driven equation is Q̈+ω²Q=z†E. The resulting ionic dielectric
contribution, returned by `ionic_dielectric`, is

\[
\Delta\epsilon_{ij}(w)=\frac{4\pi}{\Omega}
\sum_{\nu\in\mathrm{optical}}\frac{z_{i\nu}z^*_{j\nu}}
 {\omega_\nu^2-(w+i\eta)^2}.
\]

Add epsilon-infinity for the total dielectric tensor. Energies and eta are
in Eh. Positive eta is an imposed resolution; it is not a calculated decay
rate. Use eta=0 away from poles for the undamped static tensor. Only positive
optical modes may enter this spectral function; no frequency floor or
automatic acoustic filtering changes the supplied spectrum.

For a complete orthonormal mode basis, the tensor weight identity is
Σν zνzν†=Σa Z*a Z*aᵀ/Ma. A neutral Born-charge sum makes pure translations
dark. Both the raw charge-sum defect and raw acoustic force-constant defect
must be recorded before any proposed projection. These functions apply none.
At degeneracy, only complete-group tensor sums are gauge invariant; selecting
one vector does not define a unique physical intensity or derivative.

The [analytic checks](../../../tests/pbc/test_infrared.py) cover a two-atom
LO/TO splitting and Lyddane–Sachs–Teller identity, Cartesian displacement
energy, rotations, a direct oscillator resolvent, spectral-weight integrals,
complex degenerate gauges and JVP/VJP consistency. They do not establish
material convergence. Polar finite-q bands additionally require dipolar
subtraction before real-space interpolation and restoration at the target q;
the present Gamma kernel must not be used as a replacement for that step.
