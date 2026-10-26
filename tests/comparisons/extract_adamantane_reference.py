"""Vector-digitize the adamantane panel of Gali et al., Fig. 1(b).

Usage: python tests/comparisons/extract_adamantane_reference.py PAPER.pdf OUTPUT
Both theory curves are published results, not GradSCF calculations.
"""
import hashlib
import json
from pathlib import Path
import sys

import fitz
import numpy as np

pdf, output = Path(sys.argv[1]), Path(sys.argv[2])
digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
assert digest == '06876fc88abd7021b34a916a313ffd61ae62832021330bf3ab58f8a767179366'
output.mkdir(parents=True, exist_ok=True)
page = fitz.open(pdf)[2]
drawings = page.get_drawings()
curves = {}
for name, index in (('experiment', 472), ('dynamic_sf_hr', 475), ('qp_hr', 631)):
    points = []
    for item in drawings[index]['items']:
        if item[0] == 'l':
            points.extend([tuple(item[1]), tuple(item[2])])
        else:
            assert item[0] == 'c'
            a, b, c, d = np.asarray([tuple(v) for v in item[1:]])
            for t in np.linspace(0., 1., 21):
                points.append((1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d)
    xy = np.asarray(points)
    # Fig 1b major ticks 9 and 15 eV, zero-intensity baseline at y=243.806.
    energy = 9 + (xy[:, 0]-111.526)/(203.060-111.526)*6
    intensity = 243.8060302734375-xy[:, 1]
    order = np.argsort(energy)
    curves[name] = np.column_stack((energy[order], intensity[order]))
normalization = curves['experiment'][:, 1].max()
for name, xy in curves.items():
    xy[:, 1] /= normalization  # One common scale for all three curves.
    np.savetxt(output/(name+'.csv'), xy, delimiter=',', comments='',
               header='ionization_energy_ev,intensity_common_scale')
positions = []
for index in range(261, 268):
    x = drawings[index]['rect'].x0
    positions.append(9 + (x-111.526)/(203.060-111.526)*6)
np.savetxt(output/'qp_positions.csv', sorted(positions), delimiter=',', comments='', header='published_qp_marker_ev')
metadata = dict(title='Electron-vibration coupling induced renormalization in the photoemission spectrum of diamondoids',
    authors='A. Gali et al.', doi='10.1038/ncomms11327', year=2016,
    pdf_url='https://www.nature.com/articles/ncomms11327.pdf', pdf_sha256=digest,
    page=3, panel='Figure 1(b), adamantane', pymupdf_version=fitz.VersionBind,
    method='Original vector curves; cubic Bezier segments sampled with 21 points. No fitting.',
    labels=dict(experiment='Published experimental PES, attributed by Gali et al. to W. Schmidt, Tetrahedron 29, 2129 (1973).',
                qp_hr='Published G0W0 quasiparticle levels convolved with Huang-Rhys broadening; already includes static vibrational effects.',
                dynamic_sf_hr='Published dynamical electron-vibration spectral functions convolved with Huang-Rhys broadening.'),
    normalization='Original common vertical scale preserved, divided by the experimental maximum. No new energy alignment.',
    pole_warning='Vertical markers copied from panel b for position only; no spectral weights inferred.',
    attribution='Adapted/vector-redrawn from Gali et al., Nature Communications 7, 11327 (2016), CC BY 4.0. Source figure attributes experimental curve to Schmidt (1973).',
    scope='Literature illustration only, not a GradSCF calculation. Supplementary peak residuals are not electron-vibration matrix elements.')
metadata['csv_sha256'] = {name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                        for name in ('experiment.csv','qp_hr.csv','dynamic_sf_hr.csv','qp_positions.csv')}
(output/'sources.json').write_text(json.dumps(metadata, indent=2)+'\n')
print('Extracted published experimental, QP+HR and dynamical SF+HR curves.')
