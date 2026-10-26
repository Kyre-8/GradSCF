"""Extract plotted vector data from the pinned Eiguren et al. arXiv PDF.

Usage: python tests/comparisons/extract_cu111_reference.py PAPER.pdf OUTPUT_DIR
No download, OCR, fitted line shapes, or invented experimental points.
"""
import hashlib
import json
from pathlib import Path
import sys

import fitz
import numpy as np

pdf, output = Path(sys.argv[1]), Path(sys.argv[2])
digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
assert digest == '686cc14a9f458882a89cd553a86878daa7bdb28a6522f11659cd7881681b3252'
output.mkdir(parents=True, exist_ok=True)
document = fitz.open(pdf)

# PDF page 5, Fig. 1(b). Landscape coordinates after the PDF's 90-degree
# rotation: alpha2F=0 at x=406.25, 0.2 at x=721.25; Omega=0/30 meV at
# y=522.5/42.5. Solid curve is split into two paths at Omega=25.625 meV.
page = document[4]
points = []
for index in (27911, 27912):
    curve = page.get_drawings()[index]
    assert curve['dashes'] == '[] 0' and curve['type'] == 's'
    for item in curve['items']:
        assert item[0] == 'l'
        for point in item[1:]:
            x, y = point * page.rotation_matrix
            points.append(((522.5-y)/16, (x-406.25)/1575))
points = np.asarray(points)
omega = np.unique(points[:, 0])
alpha2f = np.array([points[points[:, 0] == w, 1].mean() for w in omega])
assert np.all(alpha2f >= 0)
np.savetxt(output/'alpha2f.csv', np.column_stack((omega, alpha2f)), delimiter=',',
           header='phonon_energy_mev,alpha2f_dimensionless', comments='')

# PDF page 7, Fig. 3 upper panel. Its plot box is kBT=[0,25] meV and
# Gamma=[0,40] meV. Only open-circle markers are used, not the dotted line.
page = document[6]
drawings = page.get_drawings()
rows = []
for index in range(6, 13):
    rect = drawings[index]['rect'] * page.rotation_matrix
    x, y = (rect.x0+rect.x1)/2, (rect.y0+rect.y1)/2
    thermal = (x-94.00311279296875)*25/(578.3400268554688-94.00311279296875)
    linewidth = (312.1199951171875-y)*40/(312.1199951171875-17.993000030517578)
    # The original graph supplies one representative error bar, at 125 K.
    error = (112.11800384521484-101.10200500488281)*40/(312.1199951171875-17.993000030517578) if index == 8 else np.nan
    rows.append((thermal/.08617333262145, linewidth, error))
np.savetxt(output/'experiment_linewidth.csv', rows, delimiter=',',
           header='temperature_k,intrinsic_fwhm_mev,reported_errorbar_mev', comments='')

# Fig. 3 inset: 85 individual circle markers per trace, drawn in order
# 285, 55, 160 K. Tick positions x=379.440/544.129 correspond to 500/350
# meV binding energy. Intensities have arbitrary units and no fitted baseline.
rows = []
for temperature, start in ((285, 69), (55, 154), (160, 239)):
    for index in range(start, start+85):
        marker = drawings[index]
        assert marker['type'] == 'fs' and len(marker['items']) == 4
        rect = marker['rect'] * page.rotation_matrix
        x, y = (rect.x0+rect.x1)/2, (rect.y0+rect.y1)/2
        energy = 500 - (x-379.44000244140625)*150/(544.1290283203125-379.44000244140625)
        intensity = 268.9739990234375-y
        rows.append((temperature, energy, intensity))
np.savetxt(output/'experiment_edc.csv', rows, delimiter=',',
           header='temperature_k,binding_energy_mev,intensity_plot_units', comments='')

metadata = dict(
    title='Role of bulk and surface phonons in the decay of metal surface states',
    authors='A. Eiguren et al.', doi='10.1103/PhysRevLett.88.066805',
    source_url='https://arxiv.org/pdf/cond-mat/0111029v1', pdf_sha256=digest,
    publication_year=2002, arxiv_version='cond-mat/0111029v1',
    pymupdf_version=fitz.VersionBind, extraction='Original PDF vector paths and marker centers; not original instrument files.',
    alpha2f_source='PDF p5 Fig1(b), solid curve paths 27911 and 27912; duplicate energies averaged.',
    experiment_source='PDF p7 Fig3 Cu111 upper panel open circles; inset circle markers.',
    electronic_baseline_mev=14., baseline_source='Table I: calculated electron-electron linewidth (a stated lower bound).',
    paper_tabulated_ep_zero_temperature_mev=6.6, paper_tabulated_lambda=.16,
    instrument_energy_resolution_mev=3., photon_energy_ev=21.23,
    experiment_method='He I ARUPS; published widths are fitted intrinsic Lorentzian FWHM.',
    normalization='EDCs have arbitrary intensity; peak alignment/normalization occurs only in plotting.',
    uncertainty='Only the displayed 125 K error bar is extracted. Other uncertainties are unknown, not zero.',
    limitations='Graph-derived precision; curves are not refitted to experimental widths. Not a new slab GW/DFPT calculation.',
)
metadata['csv_sha256'] = {name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                        for name in ('alpha2f.csv', 'experiment_linewidth.csv', 'experiment_edc.csv')}
(output/'sources.json').write_text(json.dumps(metadata, indent=2)+'\n')
print('Extracted',len(omega),'alpha2F samples, 7 linewidth points and 255 EDC markers.')
