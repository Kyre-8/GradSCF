# GaN compact r2SCAN: retained numerical mismatch

The native forward calculation completed in the explicitly coarse finite
model. Its independent PySCF comparison did not pass: total energy differs
by 0.46513 Eh and the maximum frontier-band difference is 0.003355 Eh.
The original inputs, arrays and failed gates remain in native.json and
pyscf.json. No figure promotes this case as an accepted numerical comparison.

The [fixed-density component diagnosis](fixed_density_components.json), run
on 2026-10-16 without a new SCF, locates a discrepancy before self-consistency.
Both packages receive the same positive-definite density matrix. Native
overlap, kinetic and hcore matrices differ by up to 0.001391, 0.09980 Eh
and 0.21465 Eh. The native finite-G mesh reconstructs band-limited AOs and
integrals; PySCF supplies analytic periodic Gaussian overlap/kinetic matrices
and real-space Gaussian AO samples. The tight Ga semicore basis is poorly
represented by this coarse mesh, so identical mesh parameters do not
establish identical discretized models.

Cross-checking on identical AO samples reduces density/gradient/tau
differences to 2.85e-14. On identical density variables, the r2SCAN XC
energy and AO-potential differences are 2.93e-13 Eh and 4.05e-14 Eh.
These checks detect no functional or potential-projection defect on that
input. They do not quantitatively decompose the earlier 0.46513-Eh
self-consistent discrepancy or establish a converged material prediction.

The diagnostic source is
[diagnose_gan_r2scan.py](../../../../tests/pbc/diagnose_gan_r2scan.py).
It ran on jy_gpu_1 with CPU0–15, CPU float64, a 600-second timeout and
a 16-GiB budget: 57.37 seconds wrapper time and 9.12 GiB peak RSS.
The executed source hashes match the current worktree and the shared
production sources in the original native record. Raw job/source records
remain in the ignored development archive. No production source was changed,
no grid scan was run and the original failed comparison remains failed.
