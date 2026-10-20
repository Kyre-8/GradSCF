"""Complete traditional XC recipes, inspected and used through public APIs.

H2/STO-3G, molecular grid level 1, atomic units, float64. A name describes
semilocal terms AND exact exchange; local-kernel availability is a separate
question from support for additional physical terms such as VV10 or PT2.
"""
import jax

jax.config.update('jax_enable_x64', True)

from gradscf import dft, gto
from gradscf.dft import xc


mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
functional = dft.custom_xc('.25*HF + .75*gga_x_pbe + gga_c_pbe',
                           name='explicit_pbe0')
print('Exact-exchange coefficient:', functional.recipe.exact_exchange.full_range)
print('Local terms:', functional.recipe.semilocal)

mf = dft.RKS(mol, xc=functional, grids_level=1,
             integral_backend='native', conv_tol=1e-11).run()
if not mf.converged:
    raise RuntimeError('PBE0 SCF did not converge.')
print('PBE0 total energy / Eh:', float(mf.e_tot))

# These calls inspect metadata; they do not run any additional calculations.
for name in ('PBE0', 'r2SCAN', 'CAM-B3LYP', 'wB97XD', 'B2PLYP'):
    metadata = xc.info(name)
    print(name, 'local backend:', metadata['backend_available'],
          'extra requirements:', metadata['extra_requirements'])

# For example, B2PLYP has an MP2 term. A local BLYP-like grid expression
# alone would omit that contribution; complete SCF consumers reject it.

# Measured on jy_gpu_1, CPU float64, JAX 0.10.0, 2026-10-20:
# Exact-exchange coefficient: 0.25
# PBE0 total energy / Eh: -1.1543306432583957
# PBE0, r2SCAN, CAM-B3LYP: local backend True; extra requirements ()
# wB97XD: local backend True; extra requirements ('dispersion',)
# B2PLYP: local backend True; extra requirements ('pt2',)
