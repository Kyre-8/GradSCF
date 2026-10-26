# WGC99 independent fixture

`libkedf_wgc_first_frame.npz` contains the first complete 16x32x48 density
and WGC nonlocal potential frame from EACcodes/libKEDF commit
`3dff53318ce7be52be5f45242ea8daf08a032866`:

- `testdata/al-lda-wgc_uneq/density_2nd`, Git blob `e2a896e222c5f5e6ec3699d0c5abbcee4949d6ef`.
- `testdata/al-lda-wgc_uneq/potential_WGC_2nd`, Git blob `e2729cb5c6b2c996c6d8ab55900fffc66e225305`.
- Configuration: `testdata/al-lda-wgc_uneq/wgc2nd_omp.stub`.

Original data were downloaded from the pinned GitHub blob endpoints. The
transfers contained at least one complete frame but did not finish the later
frames; only the complete first frame (24,576 scalar entries each) is used.
Original decimal values are preserved in float64, reshaped in Fortran order.
The stored energy is the potential file's first-frame header. Cell vectors
were converted from the stub's Angstrom units using ASE Bohr units.

This is an approximate historical regression, not machine-precision parity:
libKEDF's source uses a 5e-5 ODE integration tolerance, and its test mesh is even
and nonorthogonal. Current measured differences are 1.81e-6 Ha in energy and
6.95e-6 Ha maximum in potential. Tight odd-mesh comparisons use the independent
NumPy ODE/analytic-potential adapter instead.

Copyright (c) 2015-2016, Princeton University, Johannes M Dieterich, Emily A Carter.
BSD-3-Clause; full terms in `src/gradscf/ofdft/LICENSE.libKEDF`.
