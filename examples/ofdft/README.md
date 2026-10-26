# Orbital-free examples

Run ordinary Python scripts from the repository root with `PYTHONPATH=src`.
The [OFDFT API](../../src/gradscf/ofdft/README.md) specifies density-amplitude
representations, fixed-electron constraints and external kinetic callbacks.

For P08, run [parameter_density_response.py](parameter_density_response.py),
then [compare_parameter_density_response.py](compare_parameter_density_response.py)
and [plot_parameter_density_response.py](plot_parameter_density_response.py).
The [saved record](../../reproducibility/ad_showcase/P08_ofdft_parameter_response/README.md)
contains TF/vW analytic tests, full fixed-N density Jacobians, implicit/Taylor/FD
checks, independent Fourier/KKT results and density/energy/response maps.
It is a manufactured periodic model with explicit grid errors, not a material
prediction. The smaller [DFPT entry](../dfpt/03_ofdft_response.py) demonstrates
the same response interface with fewer lines.

[neural_kinetic.py](neural_kinetic.py) shows an external neural KEDF callback
and explicit trainable parameters. [ATLAS_REPRODUCTION.md](ATLAS_REPRODUCTION.md)
and [WGC_REPRODUCTION.md](WGC_REPRODUCTION.md) describe separate existing
material/KEDF workflows and their own evidence; P08 does not revalidate them.

The P08 [Al material extension](../../reproducibility/ad_showcase/P08_ofdft_parameter_response/large_system_scaling/README.md)
uses [large_system_response.py](large_system_response.py) for fixed-resolution
supercells and synchronized forward/JVP/VJP/HVP measurements. Run
[large_reference.py](large_reference.py) for independent DFTpy optimization
with a NumPy WGC adapter, and [large_plot.py](large_plot.py) for the saved
performance/error figures. [large_ks_reference.py](large_ks_reference.py)
keeps the finite KS comparison separate from same-model numerical agreement.
