#!/usr/bin/env python3
"""Opt-in execution of a pinned ElectronPhonon.jl synthetic self-energy oracle.

Requires Julia and an existing upstream checkout; installs/downloads nothing.
Run from the GradSCF root with PYTHONPATH=src JAX_PLATFORMS=cpu. Supply
--upstream PATH --output TMPDIR [--julia EXECUTABLE]. The output directory
receives exact upstream source extracts, an isolated Julia harness, and JSON
results/provenance. Keep upstream extracts outside the GradSCF repository.
Only archive inputs.json, results.json and provenance.json as fixtures.

The upstream numerical function is executed unchanged. Timing and state storage
are shimmed; no full-package or Wannier/material calculation is claimed.
Use --skip-gradscf for a Julia/Python-only reference execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

PIN = 'a0ecee01b7413af5566134d94cad5b15431d6565'
SOURCE_SHA256 = {
    "src/selfenergy_electron.jl": "3c5ba2d207af094c7452a0f4c6f13e8ca40cf004f511928935271f92615a8ed6",
    "src/common/utils.jl": "cf7851868badc4369039e32f97f0e8e20ce14b54d488ba769cb1b51401a2cdf0",
    "src/common/constants.jl": "1c0e37a8067b7d64839258ebd3ba7662887125d200ec4847decb7813c579390b",
    "src/common/units.jl": "b446b6850bc519f0a894c057bb26c97a5fd269b3ba370c8912fa80ce9f400414",
    "src/electron_state.jl": "41b3409ecbbdde645d9cb705e7718d1db2e6c210409ddf489cae42d5f5203e45",
    "src/EPState.jl": "ec265147dd5f63b4afdc3debfbac9a1c7605bee1e2f00b0f0ef04a5f8815ccde"
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut(text, start, end=None):
    pos = text.index(start)
    return text[pos: text.index(end, pos) if end else None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--julia', default=shutil.which('julia'))
    parser.add_argument('--skip-gradscf', action='store_true')
    args = parser.parse_args()
    if args.julia is None:
        parser.error('Julia not found; specify --julia EXECUTABLE')
    ROOT = args.upstream.resolve()
    OUT = args.output.resolve()
    JULIA = args.julia
    repository = Path(__file__).resolve().parents[2]
    if OUT == repository or repository in OUT.parents:
        parser.error('--output must be outside the repository; upstream extracts stay temporary')
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN:
        raise ValueError(f"Expected upstream {PIN}, got {commit}")
    for path, digest in SOURCE_SHA256.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Upstream SHA256 mismatch: {path}")
    source_path = ROOT / 'src/selfenergy_electron.jl'
    source = source_path.read_text()
    utils_path = ROOT / 'src/common/utils.jl'
    utils = utils_path.read_text()
    pieces = {
        'params.original.jl': cut(source, 'Base.@kwdef struct ElectronSelfEnergyParams', '# Data and buffers'),
        'compute_electron_selfen.original.jl': cut(source, '@timing "selfen_el" function compute_electron_selfen!'),
        'occ_fermion.original.jl': cut(utils, '@inline function occ_fermion(', '\n"""'),
        'occ_boson_gaussian.original.jl': cut(utils, '@inline function occ_boson(', '\n\n\n"""'),
    }
    for name, value in pieces.items():
        (OUT / name).write_text(value)

    fixtures = {
        'external_energy_ha': [-0.025, 0.035],
        'internal_energy_ha': [[-0.045, 0.008, 0.065], [-0.018, 0.026, 0.082]],
        'phonon_energy_ha': [[0.012, 0.023], [0.016, 0.028]],
        'q_weights': [0.3, 0.7],
        'chemical_potential_ha': 0.007,
        'temperature_kbt_ha': [0.0, 0.005, 0.02],
        'gaussian_width_ha': 0.017,
        'g_squared_ha2_q_internal_external_mode': [
            [[[1e-5 * (1 + q + 2*j + 3*i + mode) for mode in range(2)]
              for i in range(2)] for j in range(3)] for q in range(2)
        ],
    }
    (OUT / 'inputs.json').write_text(json.dumps(fixtures, indent=2) + '\n')

    harness = r'''# Only Base/stdlib APIs; no package installation or OffsetArrays required.
    macro timing(label, ex)
        esc(ex)
    end
    for name in ("params.original.jl", "occ_fermion.original.jl", "occ_boson_gaussian.original.jl", "compute_electron_selfen.original.jl")
        include_string(Main, read(joinpath(@__DIR__, name), String), name)
    end
    # Upstream constants.jl + units.jl: 0.01 meV in atomic Rydberg units.
    const omega_acoustic = 0.01 * (0.03674932248 * 2 * 1E-3)
    # Matches original occupation loop; NamedTuple substitutes ElectronState storage.
    function set_occupation!(el, μ, T)
        for ib in el.rng
            el.occupation[ib] = occ_fermion(el.e[ib] - μ, T)
        end
        el.occupation
    end

    energies_k_ha = [-0.025, 0.035]
    energies_kq_ha = [[-0.045, 0.008, 0.065], [-0.018, 0.026, 0.082]]
    omega_ha = [[0.012, 0.023], [0.016, 0.028]]
    weights = [0.3, 0.7]
    kbt_ha = [0.0, 0.005, 0.02]
    params = ElectronSelfEnergyParams(Tlist=2 .* kbt_ha, μ=2 * 0.007, smearing=2 * 0.017)
    el_k = (e=2 .* energies_k_ha, rng=1:2)
    total = (imsigma=zeros(Float64, 2, 1, 3),)
    per_q = []
    for q in 1:2
        # g already includes zero-point normalization: g_Ry=2*g_Ha, g²_Ry²=4*g²_Ha².
        g2 = [4e-5 * (1 + (q-1) + 2*(j-1) + 3*(i-1) + (mode-1)) for j in 1:3, i in 1:2, mode in 1:2]
        epstate = (el_k=el_k, el_kq=(e=2 .* energies_kq_ha[q], rng=1:3, occupation=zeros(3)),
                   ph=(e=2 .* omega_ha[q],), nmodes=2, g2=g2, wtq=weights[q])
        local_result = (imsigma=zeros(Float64, 2, 1, 3),)
        compute_electron_selfen!(local_result, epstate, params, 1)
        compute_electron_selfen!(total, epstate, params, 1)
        push!(per_q, [[local_result.imsigma[i,1,t] for i in 1:2] for t in 1:3])
    end
    function emit_json(x)
        if x isa AbstractVector
            print("[")
            for (i, value) in enumerate(x)
                i > 1 && print(",")
                emit_json(value)
            end
            print("]")
        else
            print(repr(x))
        end
    end
    print("{\"halfwidth_ry_q_temperature_external\":")
    emit_json(per_q)
    print(",\"halfwidth_ry_temperature_external\":")
    emit_json([[total.imsigma[i,1,t] for i in 1:2] for t in 1:3])
    print(",\"omega_acoustic_ry\":", repr(omega_acoustic), "}\n")
    '''
    (OUT / 'harness.jl').write_text(harness)
    command = [JULIA, '--startup-file=no', '--history-file=no', str(OUT / 'harness.jl')]
    julia_start = time.perf_counter()
    run = subprocess.run(command, check=True, text=True, capture_output=True)
    julia_elapsed = time.perf_counter() - julia_start
    (OUT / 'julia.stdout.json').write_text(run.stdout)
    (OUT / 'julia.stderr.txt').write_text(run.stderr)
    result = json.loads(run.stdout)
    # Halfwidth_Ry/2 is halfwidth_Ha; fullwidth_Ha is 2*halfwidth_Ha.
    result['fullwidth_ha_temperature_external'] = result['halfwidth_ry_temperature_external']
    result['fullwidth_ha_q_temperature_external'] = result['halfwidth_ry_q_temperature_external']

    reference = []
    for q in range(2):
        temperatures = []
        for kbt in fixtures['temperature_kbt_ha']:
            rows = []
            for i, external in enumerate(fixtures['external_energy_ha']):
                fullwidth = 0.0
                for j, internal in enumerate(fixtures['internal_energy_ha'][q]):
                    delta = internal - fixtures['chemical_potential_ha']
                    f = (1 / (math.exp(delta/kbt) + 1)) if kbt else (0.5 if delta == 0 else float(delta < 0))
                    for mode, omega in enumerate(fixtures['phonon_energy_ha'][q]):
                        bose = 1 / math.expm1(omega/kbt) if kbt else 0.0
                        eta = fixtures['gaussian_width_ha']
                        absorption = math.exp(-((external-internal+omega)/eta)**2) / (math.sqrt(math.pi)*eta)
                        emission = math.exp(-((external-internal-omega)/eta)**2) / (math.sqrt(math.pi)*eta)
                        g2 = fixtures['g_squared_ha2_q_internal_external_mode'][q][j][i][mode]
                        fullwidth += 2*math.pi * fixtures['q_weights'][q] * g2 * ((bose+f)*absorption+(bose+1-f)*emission)
                rows.append(fullwidth)
            temperatures.append(rows)
        reference.append(temperatures)
    errors = [abs(reference[q][t][i] - result['fullwidth_ha_q_temperature_external'][q][t][i])
              for q in range(2) for t in range(3) for i in range(2)]
    total_reference = [[sum(reference[q][t][i] for q in range(2)) for i in range(2)] for t in range(3)]
    errors += [abs(total_reference[t][i] - result['fullwidth_ha_temperature_external'][t][i]) for t in range(3) for i in range(2)]
    assert max(errors) < 1e-13, max(errors)
    result['python_independent_fullwidth_ha_temperature_external'] = total_reference
    result['max_abs_error_ha'] = max(errors)
    gradscf_metadata = None
    if not args.skip_gradscf:
        import jax
        jax.config.update("jax_enable_x64", True)
        import numpy as np
        from gradscf.gw.ep_coupling import fan_linewidth

        gradscf_start = time.perf_counter()
        gradscf_per_q = []
        betas = [1e8 if t == 0 else 1 / t for t in fixtures['temperature_kbt_ha']]
        for q, weight in enumerate(fixtures['q_weights']):
            g = np.sqrt(fixtures['g_squared_ha2_q_internal_external_mode'][q]).transpose(2, 1, 0)
            gradscf_per_q.append([
                np.asarray(fan_linewidth(
                    np.asarray(fixtures['external_energy_ha']),
                    np.asarray(fixtures['internal_energy_ha'][q]),
                    np.asarray(fixtures['phonon_energy_ha'][q]), g,
                    mu=fixtures['chemical_potential_ha'], beta=beta,
                    eta=fixtures['gaussian_width_ha'], smearing="gaussian"
                )) * weight for beta in betas
            ])
        gradscf_per_q = np.asarray(gradscf_per_q)
        gradscf_total = gradscf_per_q.sum(axis=0)
        gradscf_error = max(
            float(np.max(np.abs(gradscf_per_q - np.asarray(result['fullwidth_ha_q_temperature_external'])))),
            float(np.max(np.abs(gradscf_total - np.asarray(result['fullwidth_ha_temperature_external'])))),
        )
        result['gradscf_fullwidth_ha_q_temperature_external'] = gradscf_per_q.tolist()
        result['gradscf_fullwidth_ha_temperature_external'] = gradscf_total.tolist()
        result['gradscf_max_abs_error_ha'] = gradscf_error
        gradscf_metadata = {
            'jax_version': jax.__version__, 'numpy_version': np.__version__,
            'backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()],
            'x64_enabled': jax.config.x64_enabled,
            'beta_ha_inverse': betas, 'elapsed_seconds': time.perf_counter() - gradscf_start,
            'zero_temperature_approximation': 'beta=1e8 Ha^-1; chosen nonzero gaps and positive modes underflow to exact limiting occupations',
        }
        assert gradscf_error < 1e-13, gradscf_error
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    source_files = ['src/selfenergy_electron.jl', 'src/common/utils.jl', 'src/common/constants.jl', 'src/common/units.jl', 'src/electron_state.jl', 'src/EPState.jl']
    provenance = {
        'upstream_url': subprocess.check_output(['git', '-C', str(ROOT), 'remote', 'get-url', 'origin'], text=True).strip(),
        'upstream_commit': commit,
        'runner_command': [sys.executable, *sys.argv],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python_version': sys.version,
        'julia_elapsed_seconds': julia_elapsed,
        'gradscf': gradscf_metadata,
        'julia_version': subprocess.check_output([JULIA, '--version'], text=True).strip(),
        'command': command,
        'source_sha256': {p: sha(ROOT / p) for p in source_files},
        'extract_sha256': {name: sha(OUT / name) for name in pieces},
        'harness_sha256': sha(OUT / 'harness.jl'),
        'runner_sha256': sha(Path(__file__)),
        'inputs_sha256': sha(OUT / 'inputs.json'),
        'results_sha256': sha(OUT / 'results.json'),
        'scope': 'Execute verbatim upstream compute_electron_selfen!, parameter struct, Fermi/Bose and Gaussian functions on synthetic states. No full-package, Wannier interpolation, or physical material validation.',
        'shims': ['No-op @timing macro', 'NamedTuple epstate and imsigma storage; 1-based contiguous band indices replace OffsetArrays', 'set_occupation! reproduces original Fermi occupation loop'],
        'conventions': ['Float64 throughout', 'Gaussian delta(x)=exp(-(x/eta)^2)/(sqrt(pi)*eta)', 'T means kBT energy; Ha inputs multiplied by 2 for Ry', 'g includes zero-point normalization; squared coupling Ha² multiplied by 4 for Ry²', 'Upstream imsigma is positive halfwidth = -Im Sigma; fullwidth Gamma_Ha=2*(imsigma_Ry/2)', 'omega_acoustic_Ry=0.01*(0.03674932248*2*1e-3); all fixture modes exceed threshold', 'Validation tolerance: absolute <1e-13 Ha on all 12 per-q and 6 total fullwidth values'],
    }
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'python_max_abs_error_ha': max(errors), 'gradscf_max_abs_error_ha': result.get('gradscf_max_abs_error_ha')}, indent=2))


if __name__ == '__main__':
    main()
