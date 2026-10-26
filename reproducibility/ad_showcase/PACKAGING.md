# GW result packaging for the source checkout

This source checkout includes the selected final figures, plot arrays,
numerical summaries and bounded validation records from the periodic GW
work through 2026-10-19. The result directories retain their original
scientific scope and limitations.

Large source states and duplicate historical arrays are intentionally not
part of this Git bundle: `scgw.npz`, `scgw_dyson_diagnostics.npz`, the FFT21
`gaas_four_gw_20261019/comparison.npz`, the earlier FFT61
`gaas_four_gw_fft61_20261019/validation/before_qp_precision/comparison.npz`,
the original BerkeleyGW `raw_evidence.tar.gz`, and
`comparison_before_*.json`. These files have not been deleted or modified;
the existing local files and the execution/archive locations recorded in
the provenance remain separate from this selected source bundle.

The four `gaas_berkeleygw_matched_eta_20261018/raw/*/spectrum.dat` tables are
also retained separately. Their small input files, selected-state summaries,
exchange/Vxc and Eqp outputs remain in this bundle, alongside the final
sampled native/BerkeleyGW curve arrays and scope-specific certificates.

The original `SHA256SUMS`, `data_manifest.json`, source/provenance and
postprocessing manifests are unchanged records of the executed and archived
calculations. They may name files omitted from this Git bundle and are not
checkout-completeness manifests. Their recorded hashes identify the original
large checkpoints and historical arrays; a checkout-wide hash command based
on those original manifests will therefore report intentionally absent files.

The final FFT61 `comparison.npz` contains the saved plotting arrays, spectral
measure, raw/certified QP diagnostics and masks. It supports inspection of
the delivered figure without the full Matsubara source checkpoints. Running
the public source-state continuation again requires the separately retained
source states named by its provenance.
