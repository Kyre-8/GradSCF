# GaN: direct compact G0W0 path

The native wurtzite calculation evaluates 22 points with PBE/GTH-SZV,
GTH-PBE potentials including Ga semicore states, source mesh `(2,1,1)`
and FFT `(21,21,33)`. All 132 requested QPs pass the 1e-9 Ha root tolerance;
the largest accepted residual is 1.51e-13 Ha. Raw values and masks are in
[bands.npz](bands.npz).

The full-path GW phase takes 1523.108 s, including compilation and cache
release. The wrapper takes 1557.880 s and records 6.17 GiB peak RSS on
CPU32--47. See the [recorded budget](original_budget.json),
[figure](bands.png), [PDF](bands.pdf) and [common method/limits](../README.md).
This body-only, anisotropic two-q model is a functional demonstration,
without basis/k/FFT/frequency convergence or an experimental-gap claim.
