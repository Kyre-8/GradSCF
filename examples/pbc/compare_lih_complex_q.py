"""Independent PySCF three-cell cosine/sine gradient FD at q=(0,0,+/-1/3).

Read model definitions and the final native action, never native densities or
orbitals. Folding, each real endpoint, and final collection are separate stages
so every completed SCF/gradient result survives interruption.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import platform
import resource
from time import perf_counter

import numpy as np


STEPS = (3e-4, 1e-4)
GATES = dict(folding_energy_hartree_per_cell=1e-8, folding_relative_density=1e-7,
             native_energy_hartree=2e-6, action_hartree_per_bohr2=5e-6,
             cosine_sine_hartree_per_bohr2=5e-6, fd_step_change_hartree_per_bohr2=5e-6)
DIRECTION = np.array([[0., 0., 0.], [1., 0., 0.]])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2)+"\n")
    temporary.replace(path)


def folded_kpoints(fractional):
    folded = np.mod(np.round(np.asarray(fractional)*[1., 1., 3.], 12), 1.)
    points, groups = np.unique(folded, axis=0, return_inverse=True)
    if not np.all(np.bincount(groups) == 3):
        raise ValueError("Every three-cell k point must group exactly three primitive k points")
    return points, groups


def assert_compatible_seed(previous, current):
    """Only FFT resolution may differ for this independent warm-start."""
    old_settings = {key:value for key,value in previous["settings"].items() if key != "mesh"}
    new_settings = {key:value for key,value in current["settings"].items() if key != "mesh"}
    if old_settings != new_settings:
        raise ValueError("Independent seed settings changed beyond FFT resolution")
    for name in ("lattice_bohr", "coordinates_bohr", "fractional_kpoints", "basis_data",
                 "pseudopotential_data", "nao", "explicit_nelectron"):
        if previous[name] != current[name]:
            raise ValueError(f"Independent seed {name} changed")
    if previous.get("source_python_sha256") != current.get("source_python_sha256"):
        raise ValueError("Independent seed native model source changed")


def independent_seed_bundle(native, native_path, folder):
    """Load only checksummed densities from a completed independent oracle."""
    previous = json.loads(native_path.read_text())
    folding_path = folder/"folding.json"
    folding = json.loads(folding_path.read_text())
    if (not previous.get("complete") or not previous.get("passed")
            or not folding.get("complete") or not folding.get("passed")
            or folding["identity"]["native_sha256"] != digest(native_path)):
        raise ValueError("The independent seed's source/folding gate is not complete and matched")
    assert_compatible_seed(previous["reference"], native["reference"])
    arrays, hashes = {}, {}
    for name, repeats in (("primitive", 1), ("supercell", 3)):
        case = json.loads((folder/(name+".json")).read_text())
        state_path = folder/(name+".npz")
        if (not case.get("complete") or case["result"] != folding[name]
                or any(case["identity"].get(key) != value for key,value in folding["identity"].items())
                or digest(state_path) != case["density_sha256"]):
            raise ValueError("The independent seed checkpoint changed")
        value = np.load(state_path)["density"]
        reference = previous["reference"]
        size = repeats*reference["nao"]
        if value.shape != (len(reference["fractional_kpoints"])//repeats, size, size) or not np.isfinite(value).all():
            raise ValueError("The independent seed density has an invalid shape/value")
        arrays[name], hashes[name] = value, digest(state_path)
    arrays["provenance"] = dict(native_path=str(native_path), native_sha256=digest(native_path),
        folding_path=str(folding_path), folding_sha256=digest(folding_path), density_sha256=hashes,
        previous_fft=previous["settings"]["mesh"], current_fft=native["settings"]["mesh"],
        use="Independent PySCF densities used only as initial guesses; both new-grid SCFs are fully reconverged.")
    return arrays


def calculate(model, coordinates, repeats, *, gradient, seed=None):
    import pyscf
    from pyscf.pbc import dft, gto
    from pyscf.pbc.dft.gen_grid import UniformGrids

    started = perf_counter()
    settings = model["settings"]
    lattice = np.diag([1, 1, repeats])@np.asarray(model["lattice_bohr"])
    fractional = np.asarray(model["fractional_kpoints"])
    if repeats == 3:
        fractional, _ = folded_kpoints(fractional)
    mesh = np.asarray(settings["mesh"])*[1, 1, repeats]
    cell = gto.M(atom=list(zip(["Li", "H"]*repeats, coordinates)), a=lattice, unit="Bohr",
        basis=model["basis_data"], pseudo=dict(zip(["Li", "H"], model["pseudopotential_data"])),
        mesh=mesh, precision=settings["precision"], cart=True, spin=0, verbose=0)
    if cell.nao_nr() != model["nao"]*repeats:
        raise ValueError("Independent AO dimension differs from the recorded raw basis")
    mf = dft.KRKS(cell, kpts=cell.get_abs_kpts(fractional), exxdiv="ewald")
    mf.xc = settings["xc"]
    mf.grids = UniformGrids(cell)
    mf.grids.mesh = mesh
    mf.with_df.mesh = mesh
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 150
    energy = mf.kernel(dm0=seed)
    density, overlap = np.asarray(mf.make_rdm1()), np.asarray(mf.get_ovlp())
    nocc = 2*repeats
    energies = np.asarray(mf.mo_energy)
    gap = float(energies[:, nocc:].min()-energies[:, :nocc].max())
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock())))
    number = float(np.einsum("kij,kji->", density, overlap).real/len(overlap))
    overlap_values = np.linalg.eigvalsh(overlap)
    if not mf.converged or residual >= 2e-8 or gap <= 1e-3:
        raise RuntimeError("Independent reference is not converged and positively gapped")
    np.testing.assert_allclose(number, 4*repeats, atol=1e-9, rtol=0)
    if overlap_values.min() <= 1e-8*max(1., overlap_values.max()):
        raise RuntimeError("Independent basis is numerically linearly dependent")
    np.testing.assert_array_equal(cell.atom_charges(), model["pseudopotential_valence_charges"]*repeats)
    nuclear_gradient = np.asarray(mf.nuc_grad_method().kernel()) if gradient else None
    if gradient and not np.isfinite(nuclear_gradient).all():
        raise RuntimeError("Independent nuclear gradient is nonfinite")
    return dict(energy_hartree=float(energy), converged=True, electron_number=number,
        orbital_gradient_norm=residual, sampled_indirect_gap_hartree=gap,
        minimum_overlap_eigenvalue=float(overlap_values.min()), mesh=mesh.tolist(),
        fractional_kpoints=fractional.tolist(), coordinates_bohr=np.asarray(coordinates).tolist(),
        gradient_hartree_per_bohr=None if nuclear_gradient is None else nuclear_gradient.tolist(),
        gradient_calculated=gradient, pyscf=pyscf.__version__, elapsed_seconds=perf_counter()-started), density


def cached_calculation(folder, name, model, coordinates, repeats, gradient, identity, *, seed=None, compute=True):
    path, state_path = folder/(name+".json"), folder/(name+".npz")
    expected = dict(identity, coordinates_bohr=np.asarray(coordinates).tolist(), repeats=repeats,
                    gradient=gradient)
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("identity") != expected:
            raise ValueError("Independent endpoint identity changed; preserve the old checkpoint")
        if record.get("complete"):
            if digest(state_path) != record["density_sha256"]:
                raise ValueError("Independent endpoint density checksum changed")
            return record["result"], np.load(state_path)["density"]
    if not compute:
        raise FileNotFoundError(f"Endpoint is not complete: {path}")
    save(path, dict(complete=False, identity=expected))
    result, density = calculate(model, coordinates, repeats, gradient=gradient, seed=seed)
    temporary = state_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, density=density)
    temporary.replace(state_path)
    save(path, dict(complete=True, identity=expected, result=result, density_sha256=digest(state_path)))
    print("Saved independent checkpoint:", name, result["elapsed_seconds"], "s", flush=True)
    return result, density


def prepare(native, folder, identity, seed_bundle=None):
    model = native["reference"]
    positions, lattice = np.asarray(model["coordinates_bohr"]), np.asarray(model["lattice_bohr"])
    super_positions = (positions[None]+np.arange(3)[:, None, None]*lattice[2]).reshape(6, 3)
    path = folder/"folding.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("identity") != identity:
            raise ValueError("Independent folding identity changed")
        if saved.get("complete") and saved.get("passed"):
            return saved
    save(path, dict(complete=False, passed=False, identity=identity))
    provenance = None if seed_bundle is None else seed_bundle["provenance"]
    def seed_identity(name):
        return dict(identity, starting_density_sha256=None if provenance is None else provenance["density_sha256"][name])
    primitive, density = cached_calculation(folder, "primitive", model, positions, 1, False,
        seed_identity("primitive"), seed=None if seed_bundle is None else seed_bundle["primitive"])
    supercell, super_density = cached_calculation(folder, "supercell", model, super_positions, 3, False,
        seed_identity("supercell"), seed=None if seed_bundle is None else seed_bundle["supercell"])
    fractional = np.asarray(model["fractional_kpoints"])
    _, groups = folded_kpoints(fractional)
    folded = []
    for group in range(len(super_density)):
        indices = np.flatnonzero(groups == group)
        phases = np.exp(2j*np.pi*np.arange(3)[:, None]*fractional[indices, 2][None])/np.sqrt(3.)
        transform = np.kron(phases, np.eye(model["nao"]))
        blocks = np.zeros((3*model["nao"],)*2, dtype=complex)
        for index, source in enumerate(indices):
            start = index*model["nao"]
            blocks[start:start+model["nao"], start:start+model["nao"]] = density[source]
        folded.append(transform@blocks@transform.conj().T)
    folded = np.asarray(folded)
    errors = dict(folding_energy_hartree_per_cell=abs(supercell["energy_hartree"]/3-primitive["energy_hartree"]),
        folding_relative_density=float(np.linalg.norm(super_density-folded)/(1+np.linalg.norm(folded))),
        native_energy_hartree=abs(primitive["energy_hartree"]-model["energy_hartree"]))
    passed = bool(all(error <= GATES[name] for name, error in errors.items()))
    report = dict(complete=True, passed=passed, identity=identity, primitive=primitive, supercell=supercell,
        errors=errors, gates={name:GATES[name] for name in errors},
        super_density_sha256=digest(folder/"supercell.npz"),
        warm_start=provenance,
        scope="Independent primitive/three-cell SCF and density/energy folding; no gradient FD yet.")
    save(path, report)
    print("Independent three-cell folding:", passed, errors, flush=True)
    if not passed:
        raise RuntimeError("Independent folding failed; no FD endpoints may proceed")
    return report


def endpoint(native, folder, identity, pattern, step, sign, *, compute=True):
    folding = json.loads((folder/"folding.json").read_text())
    if folding.get("identity") != identity or not folding.get("complete") or not folding.get("passed"):
        raise ValueError("Require the matching completed independent folding gate")
    if digest(folder/"supercell.npz") != folding["super_density_sha256"]:
        raise ValueError("Independent supercell seed changed")
    model = native["reference"]
    positions, lattice = np.asarray(model["coordinates_bohr"]), np.asarray(model["lattice_bohr"])
    base = (positions[None]+np.arange(3)[:, None, None]*lattice[2]).reshape(6, 3)
    phase = np.exp(2j*np.pi*np.arange(3)*native["q_evaluated"][2])
    weights = phase.real if pattern == "cosine" else phase.imag
    displaced = base+sign*step*(weights[:, None, None]*DIRECTION).reshape(6, 3)
    label = f"h{step:.0e}_{pattern}_{'plus' if sign > 0 else 'minus'}"
    expected = dict(identity, pattern=pattern, step_bohr=step, sign=sign,
                    seed_sha256=folding["super_density_sha256"], direction=DIRECTION.tolist())
    seed = np.load(folder/"supercell.npz")["density"] if compute else None
    return cached_calculation(folder/"endpoints", label, model, displaced, 3, True, expected,
                              seed=seed, compute=compute)[0]


def collect(native, folder, identity):
    path = folder/"finite_differences.json"
    report = dict(complete=False, passed=False, identity=identity, direction=DIRECTION.tolist(), steps=[], gates=GATES,
        convention="Phi(q)u=sum_l exp(-2*pi*i*l*qz)*(gcos_prime_l+i*gsin_prime_l)/3; separate cosine/sine factors 2/3 and 2i/3.")
    save(path, report)
    matrix = np.asarray(native["matrix_real_hartree_per_bohr2"])+1j*np.asarray(native["matrix_imag_hartree_per_bohr2"])
    expected = matrix@DIRECTION.ravel()
    phase = np.exp(2j*np.pi*np.arange(3)*native["q_evaluated"][2])
    combined_actions = []
    for step in STEPS:
        derivatives = []
        for pattern in ("cosine", "sine"):
            minus = endpoint(native, folder, identity, pattern, step, -1, compute=False)
            plus = endpoint(native, folder, identity, pattern, step, 1, compute=False)
            derivatives.append((np.asarray(plus["gradient_hartree_per_bohr"])
                                -np.asarray(minus["gradient_hartree_per_bohr"])).reshape(3, 2, 3)/(2*step))
        cosine = (2/3)*np.einsum("l,lai->ai", phase.conj(), derivatives[0]).ravel()
        sine = (2j/3)*np.einsum("l,lai->ai", phase.conj(), derivatives[1]).ravel()
        combined = .5*(cosine+sine)
        errors = dict(action_hartree_per_bohr2=float(max(np.max(abs(cosine-expected)), np.max(abs(sine-expected)),
                                                       np.max(abs(combined-expected)))),
                      cosine_sine_hartree_per_bohr2=float(np.max(abs(cosine-sine))))
        report["steps"].append(dict(step_bohr=step, errors=errors,
            cosine_real=cosine.real.tolist(), cosine_imag=cosine.imag.tolist(),
            sine_real=sine.real.tolist(), sine_imag=sine.imag.tolist(),
            combined_real=combined.real.tolist(), combined_imag=combined.imag.tolist(),
            passed=bool(all(value <= GATES[key] for key, value in errors.items()))))
        combined_actions.append(combined)
        save(path, report)
    change = float(np.max(abs(combined_actions[0]-combined_actions[1])))
    report.update(complete=True, passed=all(row["passed"] for row in report["steps"]) and change <= GATES["fd_step_change_hartree_per_bohr2"],
        fd_step_change_hartree_per_bohr2=change, native_action_real=expected.real.tolist(), native_action_imag=expected.imag.tolist(),
        scope="Independent complex H-x action at the specified q/model; not independent validation of all q-mesh matrix columns.")
    save(path, report)
    print("Independent complex-q FD:", report["passed"], report["steps"], flush=True)
    if not report["passed"]:
        raise RuntimeError("Independent cosine/sine FD gate failed")
    return report


def run(native_path, output, *, stage="folding", pattern=None, step=None, sign=None,
        seed_native=None, seed_folder=None):
    """Run this example with ordinary Python parameters; no command-line parser."""
    if stage not in ("folding", "endpoint", "collect") or pattern not in (None, "cosine", "sine"):
        raise ValueError("Use a declared stage and displacement pattern")
    if sign not in (None, -1, 1):
        raise ValueError("sign must be -1 or 1")
    args = SimpleNamespace(native=Path(native_path), output=Path(output), stage=stage, pattern=pattern,
        step=step, sign=sign, seed_native=None if seed_native is None else Path(seed_native),
        seed_folder=None if seed_folder is None else Path(seed_folder))
    native = json.loads(args.native.read_text())
    q = np.asarray(native["q_evaluated"])
    if (not native.get("complete") or not native.get("passed") or native.get("asr_projected")
            or not np.allclose(q[:2], 0., atol=1e-12) or not np.isclose(abs(q[2]), 1/3, atol=1e-12)):
        raise ValueError("Require a complete raw native q=(0,0,+/-1/3) matrix")
    identity = dict(native_sha256=digest(args.native), script_sha256=digest(__file__), q_evaluated=q.tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if args.stage == "folding":
        if (args.seed_native is None) != (args.seed_folder is None):
            raise ValueError("Supply both --seed-native and --seed-folder")
        seeds = None if args.seed_native is None else independent_seed_bundle(native, args.seed_native, args.seed_folder)
        prepare(native, args.output, identity, seeds)
    elif args.stage == "endpoint":
        if args.pattern is None or args.step not in STEPS or args.sign is None:
            raise ValueError("Endpoint requires pattern, a declared step (3e-4 or 1e-4), and sign")
        endpoint(native, args.output, identity, args.pattern, args.step, args.sign)
    else:
        collect(native, args.output, identity)
    print(json.dumps(dict(stage=args.stage, elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version())), flush=True)


if __name__ == "__main__":
    run(Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x3_fft41/q_0_0_1.json"), Path("artifacts/ad_showcase/P03_complex_q_example"), stage="folding")
