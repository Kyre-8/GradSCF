# Graphene: direct finite-q G0W0 path

The native calculation evaluates 13 points on Gamma--M--K--Gamma with
PBE/GTH-SZV, a `(2,2,1)` source mesh, FFT `(25,25,97)` and the 2D finite-part
boundary. The SCF temperature is zero and the cell height is 12 Angstrom.
Complete KS clusters and T=0 touching occupations are handled in SRC.
No artificial KS gap, scissor correction or QP interpolation is used.

There are **89 accepted QPs out of 91 requests**. Point 5/band 0 and point
8/band 6 remain rejected, with residuals about -0.146915 and -0.198915 Ha.
Those values remain in [bands.npz](bands.npz) and are masked in the figure.
Because the requested occupied window is incomplete, PBE and GW share the
PBE energy zero in this panel. Small accepted residuals do not establish
that every nonlinear root belongs to a unique physical QP branch.

The full GW path takes 713.245 s; the resource wrapper takes 739.181 s and
records 6.60 GiB peak RSS on CPU48--63. The source/native fingerprints and
resource record are retained next to the data. Numerical controls, method
assumptions, missing intraband/long-wave treatment and the distinction from
converged material bands are described in the [common README](../README.md).

See the [band and residual figure](bands.png) or its [PDF](bands.pdf).
