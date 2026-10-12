"""Select a reproducible, composition-disjoint GMTKN55 v2 NNAO training sample."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

SOURCE_COMMIT = 'ccabc16c8c4677fc60c19870ae96ac25405844a2'
QUOTAS = {'2-12': dict(train=50, validation=10),
          '13-19': dict(train=65, validation=20),
          '20-29': dict(train=65, validation=15),
          '30-40': dict(train=20, validation=5)}
ATOMIC_NUMBERS = dict(H=1, C=6, N=7, O=8, F=9)
AUXILIARY_AO = dict(H=18, C=75, N=77, O=77, F=77)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def formula(row):
    return tuple(sorted(Counter(row['symbols']).items()))


def size_estimate(row):
    nh = row['symbols'].count('H'); heavy = len(row['symbols']) - nh
    nao = 4 * nh + 10 * heavy; primitive = 12 * nh + 45 * heavy
    naux = sum(AUXILIARY_AO[symbol] for symbol in row['symbols'])
    size = 8 * naux * primitive * (primitive + 1) // 2
    return dict(nao=nao, primitive_nao=primitive, naux=naux,
                df_bytes=size, df_gib=size / 2**30)


def exclusion_reason(row, max_cache_gib):
    if row.get('charge') != 0 or row.get('spin') != 0:
        return 'charge_or_spin'
    symbols = row.get('symbols', [])
    if not symbols or not set(symbols) <= ATOMIC_NUMBERS.keys():
        return 'unsupported_elements'
    if row.get('natoms') != len(symbols) or not 2 <= len(symbols) <= 40:
        return 'atom_count'
    nelectron = sum(ATOMIC_NUMBERS[symbol] for symbol in symbols)
    if nelectron % 2:
        return 'odd_electron_count'
    if row.get('nelectron') != nelectron:
        return 'inconsistent_electron_count'
    if row.get('referenced_by_retained_reaction') is not True:
        return 'not_referenced'
    try:
        coordinates = row['coords_angstrom']
        if len(coordinates) != len(symbols) or any(
            len(point) != 3 or any(not math.isfinite(x) for x in point)
            for point in coordinates
        ):
            return 'invalid_coordinates'
        if any(sum((coordinates[i][k] - coordinates[j][k])**2 for k in range(3)) < 1e-16
               for i in range(len(symbols)) for j in range(i)):
            return 'invalid_coordinates'
    except (KeyError, TypeError, ValueError):
        return 'invalid_coordinates'
    if not row.get('coordinate_signature_sha256'):
        return 'missing_coordinate_signature'
    if size_estimate(row)['df_gib'] > max_cache_gib:
        return 'df_cache_budget'
    return None


def select_structures(rows, pilot_rows, *, quotas=None, seed=0, max_cache_gib=6.):
    """Assign whole formulas before sampling at most two geometries per formula."""
    if not math.isfinite(max_cache_gib) or max_cache_gib <= 0:
        raise ValueError('max_cache_gib must be finite and positive.')
    quotas = QUOTAS if quotas is None else quotas
    identifiers = [row['structure_id'] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Duplicate structure IDs in source.')
    excluded = Counter(); eligible = []; seen_coordinates = set()
    for row in sorted(rows, key=lambda row: row['structure_id']):
        reason = exclusion_reason(row, max_cache_gib)
        signature = row.get('coordinate_signature_sha256')
        if reason is None and signature in seen_coordinates:
            reason = 'duplicate_coordinate'
        if reason is not None:
            excluded[reason] += 1
            continue
        seen_coordinates.add(signature); eligible.append(row)
    pilot_formulas = {formula(row) for row in pilot_rows}

    def order(value):
        return hashlib.sha256(json.dumps([seed, value], sort_keys=True).encode()).hexdigest()

    result = dict(train=[], validation=[], stats=dict(
        source_count=len(rows), eligible_count=len(eligible),
        eligible_formula_count=len({formula(row) for row in eligible}),
        excluded=dict(sorted(excluded.items())), bins={}))
    for label, requested in quotas.items():
        low, high = map(int, label.split('-'))
        if any(not isinstance(n, int) or n < 0 for n in requested.values()):
            raise ValueError('Quotas must be nonnegative integers.')
        groups = defaultdict(list)
        for row in eligible:
            if low <= row['natoms'] <= high:
                groups[formula(row)].append(row)
        keys = sorted(groups, key=order)
        for group in groups.values():
            group.sort(key=lambda row: order(row['structure_id']))
        validation_keys = [key for key in keys if key not in pilot_formulas][:requested['validation']]
        if len(validation_keys) != requested['validation']:
            raise ValueError(f'Insufficient unseen validation formulas in {label}.')
        validation = [groups[key][0] for key in validation_keys]
        train = []
        for layer in range(2):
            for key in keys:
                if key not in validation_keys and len(groups[key]) > layer:
                    train.append(groups[key][layer])
            if len(train) >= requested['train']:
                break
        train = train[:requested['train']]
        if len(train) != requested['train']:
            raise ValueError(f'Insufficient train geometries in {label}; do not relax selection rules.')
        result['stats']['bins'][label] = dict(eligible_count=sum(map(len, groups.values())),
            eligible_formula_count=len(groups), train_count=len(train), validation_count=len(validation))
        for split, selected in [('train', train), ('validation', validation)]:
            result[split].extend(dict(row, name=row['structure_id'], split=split,
                formula_group=''.join(f'{element}{count}' for element, count in formula(row)),
                size_estimate=size_estimate(row)) for row in selected)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('/Volumes/TF/GradSCF_GMTKN55'))
    parser.add_argument('--output-dir', type=Path,
        default=Path('artifacts/nnao-stage2-20261007/selection'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    source = args.dataset_root / 'data/v2/structures.jsonl'
    pilot = args.dataset_root / 'selections/hf_pilot_20261007/structures.jsonl'
    provenance = args.dataset_root / 'metadata/sources.json'
    version = json.loads(provenance.read_text())['versions']['v2']
    if version['commit'] != SOURCE_COMMIT:
        raise ValueError('GMTKN55 v2 source commit differs from the approved snapshot.')
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = select_structures(read(source), read(pilot), seed=args.seed)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing selection: {output}')
    output.mkdir(parents=True, exist_ok=True)
    all_rows = result['train'] + result['validation']
    for name, rows in [('train', result['train']), ('validation', result['validation']), ('all', all_rows)]:
        (output / f'{name}.jsonl').write_text(''.join(
            json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows))
    with (output / 'systems.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['structure_id', 'split', 'subset',
            'formula_group', 'natoms', 'nao', 'primitive_nao', 'naux', 'df_bytes', 'df_gib'])
        writer.writeheader()
        writer.writerows(dict(structure_id=row['structure_id'], split=row['split'],
            subset=row['subset'], formula_group=row['formula_group'], natoms=row['natoms'],
            **row['size_estimate']) for row in all_rows)
    manifest = dict(source_version='v2', source_commit=SOURCE_COMMIT,
        source_repository=version['repository'], source_structures_sha256=sha256(source),
        source_metadata_sha256=sha256(provenance), pilot_structures_sha256=sha256(pilot),
        selector_sha256=sha256(__file__), seed=args.seed, quotas=QUOTAS,
        fresh_model_initialization_required=True, max_geometries_per_formula=2,
        validation_excludes_pilot_formulas=True, reaction_labels_used=False,
        method='RHF', basis='szp663_direct', core_primitives=6, cartesian=False,
        max_df_cache_gib=6., estimate_model=dict(hydrogen_nao=4, hydrogen_primitive_nao=12,
            heavy_nao=10, heavy_primitive_nao=45, auxiliary_nao=AUXILIARY_AO,
            df_dtype='float64', df_layout='naux x nprimitive*(nprimitive+1)/2',
            auxiliary_basis='def2-universal-jkfit',
            warning='Cache bytes exclude integral preparation and SCF/backward temporary memory.'),
        stats=result['stats'], train_structure_ids=[row['structure_id'] for row in result['train']],
        validation_structure_ids=[row['structure_id'] for row in result['validation']],
        total_estimated_df_bytes=sum(row['size_estimate']['df_bytes'] for row in all_rows),
        maximum_estimated_df_bytes=max(row['size_estimate']['df_bytes'] for row in all_rows))
    (output / 'selection.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (output / 'README.md').write_text(
        '# NNAO 第二阶段小样本训练选择\n\n'
        'GMTKN55 v2 固定快照；200 个训练结构、50 个验证结构，模型从头初始化。'
        '训练目标为收敛 RHF 总能量的变分最小化，不使用反应能参考标签。\n\n'
        '筛选 CHNOF、中性、spin=0、偶电子、2–40 原子、被保留反应引用且无重合核的有限坐标。'
        '精确坐标哈希去重；每个化学式最多两个几何，整式划分训练和验证，验证排除 pilot 已见化学式。'
        '该划分不保证反应连通分组隔离，不用于完整 GMTKN55 反应能基准声明。\n\n'
        '基组为全电子球谐 szp663_direct/core6；DF 辅助基组 def2-universal-jkfit。'
        '单个 primitive packed DF 缓存估计上限 6 GiB；内存估计不包含临时张量和求导。'
        '训练应按内存预算逐结构加载，不能将全部缓存常驻 GPU。\n\n'
        '| 原子数 | Train | Validation |\n|---|---:|---:|\n'
        '| 2–12 | 50 | 10 |\n| 13–19 | 65 | 20 |\n'
        '| 20–29 | 65 | 15 |\n| 30–40 | 20 | 5 |\n\n'
        'selection.json 记录源文件和选择脚本哈希、seed=0、筛除原因与尺寸估算。'
        '本步骤只进行结构选择，未运行 SCF 或训练。\n')
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / 'SHA256SUMS').write_text(''.join(f'{sha256(path)}  {path.name}\n' for path in files))
    print(json.dumps(dict(train=len(result['train']), validation=len(result['validation']),
        stats=result['stats'], total_df_gib=manifest['total_estimated_df_bytes'] / 2**30,
        maximum_df_gib=manifest['maximum_estimated_df_bytes'] / 2**30), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
