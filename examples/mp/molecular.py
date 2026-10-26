"""Canonical molecular MP2/MP3; GradSCF only, energies in Hartree."""
import jax

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, mp

mol = gto.M(atom="O 0 0 0; H 0 .75 .58; H 0 -.75 .58",
            basis="6-31g*", cart=True)
mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_grad=1e-10).run()

pt2 = mp.MP2(mf, frozen=1).run()
pt3 = mp.MP3(mf, frozen=1).run()
print("RHF energy       % .12f" % mf.e_tot)
print("MP2 correction   % .12f" % pt2.e2)
print("MP3 correction   % .12f" % pt3.e3)
print("MP3 total        % .12f" % pt3.e_tot)

# DF-MP2 streams occupied-index integral slices and can omit saved amplitudes.
dfmf = scf.RHF(mol, conv_tol=1e-12, conv_tol_grad=1e-10).density_fit().run()
dfpt = mp.MP2(dfmf, frozen=1, with_t2=False).run()
print("DF-MP2 total     % .12f" % dfpt.e_tot)

# An open-shell reference dispatches to UMP2.
radical = gto.M(atom="H 0 0 0; H 0 0 .85; H 0 0 1.9", basis="sto-3g", spin=1)
umf = scf.UHF(radical, conv_tol=1e-13, conv_tol_grad=1e-11, max_cycle=150).run()
upt = mp.MP2(umf).run()
print("UMP2 total       % .12f" % upt.e_tot)

# Example output (CPU, JAX float64, native integrals, Cartesian basis):
# RHF energy       -76.010721112637
# MP2 correction   -0.185314572684
# MP3 correction   -0.005996691699
# MP3 total        -76.202032377021
# DF-MP2 total     -76.196039459056
# UMP2 total       -1.562947444299
