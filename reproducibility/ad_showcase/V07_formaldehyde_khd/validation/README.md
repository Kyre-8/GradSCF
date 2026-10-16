# V07 validation evidence

These text files preserve remote stdout/stderr. Matching `.exit` files record
the actual process exit code. Files ending in `.txt` were copied unchanged from
the corresponding remote `.log`; this avoids the repository-wide log ignore.

| Record | Outcome |
| --- | --- |
| `red` | Missing API: 8 failures, exit 1 |
| `green` | Initial KHD plus old FC/HT regression suite: 24 passed, exit 0 |
| `resolvent` | Independent padded harmonic-basis complex resolvent: exit 0 |
| `molecular` | HCHO N=24, three damping values and grid gates: exit 0 |
| `input-red` | Six explicit input-type rejection cases failed, exit 1 |
| `input-green` | Final KHD plus old FC/HT regression suite: 30 passed, exit 0 |
| `input-parity` | Final guard versus stored valid-input calculation: exit 0 |
| `plot` | Saved-data figure generation: exit 0; empty stdout |

The source archive and `calculation_runtime.json` preserve the executed
scientific helper before the input guard update. The final helper rejects
complex energy/frequency inputs and non-real/nonscalar Gamma. No valid-input
formula was changed. `artifacts/ad_showcase/provenance/V07_formaldehyde_khd/validation/verify_input_guard.py` is the exact one-off audit script;
it reads the large raw vertices retained in the remote snapshot. Reproduction
commands and both source versions are recorded in `../source_manifest.json`.

This is a component check, not a complete GradSCF integration test. The HCHO
electronic model is inherited from X11/X12; the independent resolvent oracle
uses a synthetic two-mode model, not another HCHO electronic calculation.
