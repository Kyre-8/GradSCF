# BSE equations, software and validation references

1. X. Blase, I. Duchemin, D. Jacquemin, and P.-F. Loos,
   “The Bethe–Salpeter Equation Formalism: From Physics to Chemistry,”
   *J. Phys. Chem. Lett.* **11**, 7371–7382 (2020).
   [doi:10.1021/acs.jpclett.0c01875](https://doi.org/10.1021/acs.jpclett.0c01875),
   [author manuscript](https://arxiv.org/abs/2006.09440).
   Equations 17–23 define the static kernel, singlet/triplet factors and
   transition-space integral ordering used here.
2. L. Hedin, *Phys. Rev.* **139**, A796 (1965),
   [doi:10.1103/PhysRev.139.A796](https://doi.org/10.1103/PhysRev.139.A796).
   GW/RPA screening context; the auxiliary-factor convention also follows the
   existing GradSCF GW module documentation.
3. [QuAcK](https://github.com/pfloos/QuAcK), commit
   `2236bfcda24ff0971358f5107636b506dd201bb1` (2026-09-02).
   The unmodified Fortran routines `phRLR_A.f90`, `phRLR_B.f90`,
   `RGW_phBSE_static_kernel_A.f90` and `RGW_phBSE_static_kernel_B.f90` were
   compiled and run as an external kernel oracle.
   QuAcK's KA is a **screening correction**, not full W or the complete BSE A.
   Independent NumPy full direct-RPA poles and excitation densities supply its
   input. Adding its bare resonant block and KA is compared with GradSCF's
   auxiliary-inverse formulation. Source hashes, compiler and commands are in
   `tests/bse/data/quack_static_seed83.json` (original TDA run) and
   `tests/bse/data/quack_full_static_seed83.json` (A/B/full-BSE comparison).
   The full-BSE reference eigenvalues use NumPy on the doubled matrix formed
   from those actual Fortran blocks; a QuAcK eigensolver was not executed.
4. [MOLGW BSE tutorial](https://www.molgw.org/tuto_bse/) and
   [VOTCA-XTP architecture](https://www.votca.org/xtp/Architecture.html).
   Consulted for molecular workflow and matrix-free organization only. No
   MOLGW or VOTCA executable was run during this implementation.
5. [PySCF user documentation](https://pyscf.org/user/tddft.html).
   Installed PySCF 2.9.0 supplies the HF/CIS and HF/TDHF unscreened-limit oracles, not a BSE
   oracle. Its version has no `pyscf.gw.bse`; the newer online BSE code was
   inspected during planning but is not the executed acceptance reference.
6. E. Monino and P.-F. Loos, “Spin-Conserved and Spin-Flip Optical Excitations
   from the Bethe–Salpeter Equation Formalism,” *J. Chem. Theory Comput.*
   **17**, 2852–2867 (2021),
   [doi:10.1021/acs.jctc.1c00074](https://doi.org/10.1021/acs.jctc.1c00074).
   Spin-resolved GW screening and unrestricted spin-conserving A/B blocks.
   This implementation does not include the paper's spin-flip or dynamical
   extensions. Tests use an independent NumPy contraction oracle and the
   restricted singlet/triplet limit; no unrestricted external BSE executable
   was used for this extension.

The production JAX BSE code is a new implementation of the stated equations.
No QuAcK, MOLGW, VOTCA or newer PySCF BSE source is vendored or executed by it.
The optional comparison script downloads pinned QuAcK sources to a temporary
cache and checks their SHA256 before compilation. Normal pytest uses small
saved numerical reference arrays and does not require network access or Fortran.

Agreement with a same-input static kernel is an implementation check. It does
not establish equality of different GW starting points, screening prescriptions,
basis sets, or dynamic/static approximations, nor physical accuracy against
FCI or experimental excitation energies.
