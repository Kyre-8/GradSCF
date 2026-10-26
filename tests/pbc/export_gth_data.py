"""Rebuild the offline GTH snapshot from the pinned PySCF 2.9.0 distribution.

Run from the repository root; --check verifies byte-for-byte reproducibility.
PySCF is an export/test dependency only, never a runtime basis dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import pyscf
from pyscf.pbc.gto import basis, pseudo


def normalize(name):
    return re.sub(r'[-_\s]', '', name.lower())


def headers(path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r'^\s*([A-Z][a-z]?)\s+([^#\s].*)$', line)
        if match:
            result.setdefault(match[1], []).append(match[2].split())
    return result


def export():
    if pyscf.__version__ != '2.9.0':
        raise RuntimeError('This snapshot is pinned to PySCF 2.9.0')
    root = Path(pyscf.__file__).parent/'pbc'/'gto'
    data = dict(basis={}, pseudo={}, basis_valence={}, aliases={}, pseudo_variants={}, unavailable={})
    checksums = {}
    # Prefer unsuffixed families for generic CP2K aliases. Explicit q aliases
    # are per-element, since e.g. Na-q9 and Rh-q9 live in different files.
    paths = sorted((root/'basis').glob('*.dat'), key=lambda p: (bool(re.search(r'-q\d+$', p.stem)), p.name))
    for path in paths:
        family = path.stem
        data['basis'][family], data['basis_valence'][family] = {}, {}
        checksums[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for symbol, entries in headers(path).items():
            if len(entries) != 1:
                raise ValueError(f'Ambiguous basis block: {path.name}/{symbol}')
            labels = entries[0]
            try:
                data['basis'][family][symbol] = basis.load(str(path), symbol)
            except ValueError as error:
                # These two upstream 2.9.0 blocks have incomplete rows. Do not
                # invent contraction coefficients or silently omit other errors.
                if ((family, symbol) not in {('gth-aug-tzvp', 'O'), ('gth-aug-tzv2p', 'O')}
                        or str(error) != 'Basis data incomplete'):
                    raise
                data['unavailable'][f'{family}/{symbol}'] = str(error)
                continue
            charges = {int(m[1]) for label in labels if (m := re.search(r'-q(\d+)$', label, re.I))}
            if len(charges) > 1:
                raise ValueError(f'Conflicting valence labels: {path.name}/{symbol}')
            q = next(iter(charges), None)
            data['basis_valence'][family][symbol] = q
            for label in labels:
                data['aliases'].setdefault(normalize(label), {}).setdefault(symbol, family)
    for family in ('gth-pade', 'gth-pbe', 'gth-blyp'):
        path = root/'pseudo'/f'{family}.dat'
        checksums[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        data['pseudo'][family], data['pseudo_variants'][family] = {}, {}
        for symbol, entries in headers(path).items():
            default = pseudo.load(family, symbol)
            data['pseudo'][family][symbol] = default
            variants = {}
            for labels in entries:
                for label in labels:
                    match = re.search(r'-q(\d+)$', label, re.I)
                    if match and int(match[1]) != sum(default[0]):
                        variants[match[1]] = pseudo.load(f'{family}-q{match[1]}', symbol)
            if variants:
                data['pseudo_variants'][family][symbol] = variants
    data['provenance'] = dict(project='PySCF', version=pyscf.__version__,
        url='https://github.com/pyscf/pyscf/tree/v2.9.0/pyscf/pbc/gto', sha256=checksums)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[2]/'src/gradscf/integrals/periodic/gth_data.json'
    result = export()
    content = json.dumps(result, sort_keys=True, separators=(',', ':'))+'\n'
    if args.check:
        if destination.read_text() != content:
            raise SystemExit('GTH snapshot differs from pinned source data')
    else:
        # Keep all previously exported default entries numerically unchanged.
        old = json.loads(destination.read_text())
        for kind in ('basis', 'pseudo'):
            for family, elements in old[kind].items():
                for element, values in elements.items():
                    if values != result[kind][family][element]:
                        raise ValueError(f'Existing data changed: {family}/{element}')
        destination.write_text(content)
    print(f"{len(result['basis'])} basis families; {sum(map(len, result['basis'].values()))} element/family entries")
