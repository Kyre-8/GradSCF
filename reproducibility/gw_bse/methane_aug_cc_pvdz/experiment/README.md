# Experimental methane photoabsorption data

Downloaded from the MPI-Mainz UV/VIS Spectral Atlas on 2026-09-30.
The raw two-column files are retained locally and are not committed. Exact
download URLs and SHA256 checksums are in `sources.json`. This comparison uses total absorption
cross sections, not photoionization-only or fragment-production cross sections.

## Obtain the reference tables before plotting

Download each `datasets[].url` from `sources.json` into this directory using
its exact `datasets[].file` name. Verify each file against `datasets[].sha256`
before running either methane experimental-comparison plotting script.
The following standard-library snippet runs from the repository root and
keeps normal HTTPS certificate verification enabled. Existing local files
are verified without downloading them again:

```sh
python - <<'PY'
from pathlib import Path
import hashlib
import json
import urllib.request

folder = Path('reproducibility/gw_bse/methane_aug_cc_pvdz/experiment')
sources = json.loads((folder / 'sources.json').read_text())
for source in sources['datasets']:
    path = folder / source['file']
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(source['url'], timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != source['sha256']:
        raise ValueError(f'SHA256 mismatch: {path}')
    if not path.exists():
        path.write_bytes(data)
    print(f'Verified {path}')
PY
```

If HTTPS validation fails, obtain the same table through a trusted,
certificate-valid source and verify its recorded checksum; the reproduction
instructions do not disable TLS validation. Keep downloaded raw tables out
of commits.

## Sources and units

1. K. Kameta, N. Kouchi, M. Ukai, Y. Hatano, *Photoabsorption,
   photoionization, and neutral-dissociation cross sections of simple
   hydrocarbons in the vacuum ultraviolet range*, Journal of Electron
   Spectroscopy and Related Phenomena **123**, 225-238 (2002).
   DOI: https://doi.org/10.1016/S0368-2048(02)00022-1.
   Atlas temperature: 298 K. Numerical span: 52.054-124.63 nm,
   approximately 9.948-23.818 eV; 3077 rows. The atlas identifies these as
   author-supplied data communicated in January 2003, measured with a double
   ionization chamber and synchrotron radiation. Main comparison reference.
2. J. A. R. Samson, G. N. Haddad, T. Masuoka, P. N. Pareek, D. A. L. Kilcoyne,
   *Ionization yields, total absorption, and dissociative photoionization
   cross sections of CH4 from 110-950 A*, J. Chem. Phys. **90**, 6925-6931 (1989).
   DOI: https://doi.org/10.1063/1.456267.
   Atlas temperature: 298 K. Archived table spans 10-95 nm, 87 rows;
   the title's wavelength range and the archive's span are not identical.
   Independent high-energy overlap; not spliced with the main dataset.
3. L. C. Lee and C. C. Chiang, *Fluorescence yield from photodissociation
   of CH4 at 1060-1420 A*, J. Chem. Phys. **78**, 688-691 (1983).
   DOI: https://doi.org/10.1063/1.444812.
   Atlas temperature: 293 K, resolution 0.2 nm. The archived absorption
   curve was digitized from Figure 1 (129 rows), rather than supplied as an
   original numerical table. Shown as a secondary low-energy reference.

Atlas citation: H. Keller-Rudek, G. K. Moortgat, R. Sander, R. Sörensen,
*The MPI-Mainz UV/VIS spectral atlas of gaseous molecules of atmospheric
interest*, Earth Syst. Sci. Data **5**, 365-373 (2013).
https://doi.org/10.5194/essd-5-365-2013.

Atlas format documentation:
https://uvvis.mpch-mainz.gwdg.de/uvvis/index.html.
Column 1 is wavelength in nm; column 2 is absorption cross section in
cm^2/molecule. For these VUV measurements we use photon energy hc/lambda:
E[eV] = 1239.841984... / lambda[nm]; sigma[Mb] = 1e18 sigma[cm^2].
Cross section is an observable evaluated at that photon energy, so no
wavelength-to-energy density Jacobian is applied. No air-index correction
or spectral-axis calibration adjustment is made.

The vertical line at 12.61 eV is the evaluated experimental ionization onset
(12.61 +/- 0.01 eV), from the NIST Chemistry WebBook:
https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Mask=20.
It is not the vertical ionization energy or a predicted BSE threshold.

The public atlas server presented an expired TLS certificate during retrieval.
Verification was disabled only for those individual public-data downloads;
no system trust settings were changed. Checksums provide reproducible local
file integrity, not an independent authentication of the remote server.
