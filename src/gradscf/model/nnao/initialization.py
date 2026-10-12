"""Reference exponent pools and contraction templates for NNAO initialization."""
from functools import lru_cache
from importlib.resources import files
import json
import numpy as np

@lru_cache(maxsize=1)
def _templates():
    return json.loads(files('gradscf.model.nnao').joinpath('data/szp3.json').read_text())['elements']


@lru_cache(maxsize=3)
def _direct_templates(basis_family):
    from copy import deepcopy
    if basis_family not in {'szp3_direct','szp442_direct','szp663_direct'}:
        raise ValueError('Unknown all-electron direct basis family.')
    if basis_family=='szp663_direct':
        data=deepcopy(_direct_templates('szp442_direct'))
        for element in data.values():
            for shell in element['shells']:
                if shell['role']=='core':continue
                a=shell['exponents'];c=shell['coefficients']
                target=6 if shell['role'].startswith('valence_') else (3 if shell['l']==2 else 2)
                # Preserve old exponents, add tight/diffuse endpoints, then
                # bisect the widest logarithmic gap if one more is needed.
                additions=[2*max(a),min(a)/2]
                while len(a)<target:
                    if additions:extra=additions.pop(0)
                    else:
                        ordered=np.sort(a);i=int(np.argmax(np.diff(np.log(ordered))))
                        extra=float(np.sqrt(ordered[i]*ordered[i+1]))
                    a.append(extra);c.append(0.)
        return data
    data=deepcopy(_templates())
    if basis_family=='szp442_direct':
        for element in data.values():
            # Expand only existing p-block sp+d templates; do not add new
            # angular shells to H/He or the s-block element templates.
            if not any(s['role']=='valence_p' for s in element['shells']):continue
            for shell in element['shells']:
                if shell['role'] in {'valence_s','valence_p'} or (shell['role']=='polarization' and shell['l']==2):
                    shell['exponents'].append(min(shell['exponents'])/2.)
                    shell['coefficients'].append(0.)
    return data


def _direct_slot(shell):
    slot={'valence_s':0,'valence_p':1}.get(shell['role'],-1)
    if shell['role']=='polarization' and len(shell['exponents'])>1:
        slot=shell['l']
    return slot
