# Configuration-interaction examples

Run ordinary scripts from the repository root with `PYTHONPATH=src`.

| Example | Scope |
| --- | --- |
| [Restricted CI](restricted_ci.py) | Restricted configuration-interaction APIs. |
| [Spin CID](cid_spin.py) | Spin-resolved doubles configuration interaction. |
| [Water CIS probes](cis_h2o_gradients.py) | X06: squared transition-moment sensitivities to relaxed geometry, normalized amplitudes and fixed-X reference rotations. |
| [Independent water probe checks](compare_cis_h2o_probes.py) | Fresh tracked PySCF geometry states and independent contractions for the two fixed-state partials. |
| [Water sensitivity plots](plot_cis_h2o_probes.py) | Recorded derivative heatmaps and directional finite-difference errors. |

The [X06 record](../../reproducibility/ad_showcase/X06_water_cis_probes/README.md)
states normalization, gauge, parameter maps, error gates and provenance.
Amplitude/reference heatmaps use the recorded MO frame; their components are
covariant coordinates of a gauge-invariant scalar, not individually invariant
observables. The three derivative maps do not automatically sum to a nuclear
response. Related CIS spectra, densities and geometry examples are indexed in
[tdscf](../tdscf/README.md).
