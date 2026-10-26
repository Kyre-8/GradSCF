from __future__ import annotations

import ast
import copy
import re
from functools import lru_cache
from pathlib import Path


_MAPSPDF = {
    "S": 0,
    "P": 1,
    "D": 2,
    "F": 3,
    "G": 4,
    "H": 5,
    "I": 6,
    "J": 7,
    "K": 8,
}
_BASIS_SET_DELIMITER = re.compile(r"# *BASIS SET.*\n|END\n")
_POPLE_ALIAS = {
    "321++g": Path("pople-basis/3-21++G.dat"),
    "321++g*": Path("pople-basis/3-21++Gs.dat"),
    "321++gs": Path("pople-basis/3-21++Gs.dat"),
    "321g": Path("pople-basis/3-21G.dat"),
    "321g*": Path("pople-basis/3-21Gs.dat"),
    "321gs": Path("pople-basis/3-21Gs.dat"),
    "431g": Path("pople-basis/4-31G.dat"),
    "631++g": Path("pople-basis/6-31++G.dat"),
    "631++g*": Path("pople-basis/6-31++Gs.dat"),
    "631++gs": Path("pople-basis/6-31++Gs.dat"),
    "631++g**": Path("pople-basis/6-31++Gss.dat"),
    "631++gss": Path("pople-basis/6-31++Gss.dat"),
    "631+g": Path("pople-basis/6-31+G.dat"),
    "631+g*": Path("pople-basis/6-31+Gs.dat"),
    "631+gs": Path("pople-basis/6-31+Gs.dat"),
    "631+g**": Path("pople-basis/6-31+Gss.dat"),
    "631+gss": Path("pople-basis/6-31+Gss.dat"),
    "6311++g": Path("pople-basis/6-311++G.dat"),
    "6311++g*": Path("pople-basis/6-311++Gs.dat"),
    "6311++gs": Path("pople-basis/6-311++Gs.dat"),
    "6311++g**": Path("pople-basis/6-311++Gss.dat"),
    "6311++gss": Path("pople-basis/6-311++Gss.dat"),
    "6311+g": Path("pople-basis/6-311+G.dat"),
    "6311+g*": Path("pople-basis/6-311+Gs.dat"),
    "6311+gs": Path("pople-basis/6-311+Gs.dat"),
    "6311+g**": Path("pople-basis/6-311+Gss.dat"),
    "6311+gss": Path("pople-basis/6-311+Gss.dat"),
    "6311g": Path("pople-basis/6-311G.dat"),
    "6311g*": Path("pople-basis/6-311Gs.dat"),
    "6311gs": Path("pople-basis/6-311Gs.dat"),
    "6311g**": Path("pople-basis/6-311Gss.dat"),
    "6311gss": Path("pople-basis/6-311Gss.dat"),
    "631g": Path("pople-basis/6-31G.dat"),
    "631g*": Path("pople-basis/6-31Gs.dat"),
    "631gs": Path("pople-basis/6-31Gs.dat"),
    "631g**": Path("pople-basis/6-31Gss.dat"),
    "631gss": Path("pople-basis/6-31Gss.dat"),
}


def _snapshot_root() -> Path:
    return Path(__file__).with_name("pyscf_basis_snapshot")


@lru_cache(maxsize=1)
def _snapshot_aliases():
    """Read pinned alias data without importing the upstream PySCF module."""
    def literal(node):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in {"join", "os.path.join"}:
            return str(Path(*(literal(arg) for arg in node.args)))
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(literal(item) for item in node.elts)
        return ast.literal_eval(node)
    tree = ast.parse((_snapshot_root()/"__init__.pydata").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALIAS" for t in node.targets):
            return {ast.literal_eval(k): literal(v) for k, v in zip(node.value.keys, node.value.values)}
    raise RuntimeError("Pinned basis snapshot has no alias table")


def _python_basis(path, symbol):
    """Load literal element data, including kappa labels, without executing code."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"Basis block for symbol {symbol!r} not found in {path.name}")


def _std_symbol(symbol: str) -> str:
    letters = "".join(ch for ch in str(symbol) if ch.isalpha())
    if not letters:
        raise ValueError(f"Invalid atomic symbol {symbol!r}.")
    return letters[0].upper() + letters[1:].lower()


def _format_basis_name(basisname: str) -> str:
    return str(basisname).lower().replace("-", "").replace("_", "").replace(" ", "")


def _is_pople_basis(basisname: str) -> bool:
    return (
        basisname.startswith("631")
        or basisname.startswith("321")
        or basisname.startswith("431")
    )


@lru_cache(maxsize=1)
def _normalized_dat_file_map() -> dict[str, Path]:
    root = _snapshot_root()
    mapping: dict[str, Path] = {}
    for path in root.rglob("*.dat"):
        rel = path.relative_to(root)
        key = _format_basis_name(path.stem)
        mapping.setdefault(key, rel)
    return mapping


def _parse_pople_basis_paths(basisname: str, symbol: str) -> tuple[Path, ...]:
    if "(" in basisname:
        mbas = basisname[: basisname.find("(")]
        extension = basisname[basisname.find("(") + 1 : basisname.find(")")]
    else:
        mbas = basisname
        extension = ""

    basename = (mbas[0] + "-" + mbas[1:].upper()).replace("+", "").replace("*", "")
    pathtmp = "pople-basis/" + basename + "-polarization-%s.dat"

    def convert(spec: str) -> list[Path]:
        if len(spec) == 0:
            return []
        if spec[0].isalpha():
            return [Path(pathtmp % spec[0])] + convert(spec[1:])
        return [Path(pathtmp % spec[:2])] + convert(spec[2:])

    if mbas not in _POPLE_ALIAS:
        raise KeyError(f"Unsupported Pople basis alias {mbas!r}.")
    main = _POPLE_ALIAS[mbas]

    sym = _std_symbol(symbol)
    if sym in ("H", "He"):
        if "," in extension:
            return tuple([main] + convert(extension.split(",")[1]))
        return (main,)
    if "," in extension:
        return tuple([main] + convert(extension.split(",")[0]))
    return tuple([main] + convert(extension))


def _search_basis_block(raw_text: str, symbol: str) -> list[str]:
    sym = _std_symbol(symbol)
    chunks = re.split(_BASIS_SET_DELIMITER, raw_text)
    for chunk in chunks:
        first = chunk.split(None, 1)
        if not first:
            continue
        if first[0] == sym:
            return chunk.splitlines()
        if first[0].startswith("#"):
            lines = chunk.splitlines()
            for i, line in enumerate(lines):
                if not line:
                    continue
                stripped = line.lstrip()
                if not stripped or stripped[0] == "#":
                    continue
                if line.split(None, 1)[0] == sym:
                    return lines[i:]
                break
    raise KeyError(f"Basis block for symbol {sym!r} not found.")


def _parse_nwchem_block(lines: list[str]) -> list[list[object]]:
    basis_parsed: list[list[list[object]]] = [[] for _ in range(max(_MAPSPDF.values()) + 1)]
    key: str | None = None
    for raw_line in lines:
        line = raw_line.split("#")[0].strip()
        line_upper = line.upper()
        if not line or line_upper.startswith("END") or line_upper.startswith("BASIS"):
            continue
        if line[0].isalpha():
            parts = line.split()
            key = parts[0].upper() if len(parts) == 1 else parts[1].upper()
            if key == "SP":
                basis_parsed[0].append([0])
                basis_parsed[1].append([1])
            elif key in _MAPSPDF:
                basis_parsed[_MAPSPDF[key]].append([_MAPSPDF[key]])
            else:
                raise ValueError(f"Unsupported shell label {key!r} in vendored basis parser.")
            continue

        data = [float(x.replace("D", "e")) for x in line.split()]
        if key is None:
            raise ValueError("Encountered primitive line before shell header.")
        if key == "SP":
            basis_parsed[0][-1].append([data[0], data[1]])
            basis_parsed[1][-1].append([data[0], data[2]])
        else:
            basis_parsed[_MAPSPDF[key]][-1].append(data)
    basis_sorted = [segment for per_l in basis_parsed for segment in per_l]
    if not basis_sorted:
        raise ValueError("No basis data parsed from vendored snapshot block.")
    return basis_sorted


@lru_cache(maxsize=256)
def _load_basis_from_snapshot(basisname: str, symbol: str) -> list[list[object]]:
    root = _snapshot_root()
    name = _format_basis_name(basisname)

    aliases = _snapshot_aliases()
    if name in aliases:
        target = aliases[name]
        targets = (target,) if isinstance(target, str) else target
        relpaths = tuple(Path(t) if t.endswith(".dat") else Path(t.replace(".", "/")+".pydata") for t in targets)
    elif _is_pople_basis(name):
        relpaths = _parse_pople_basis_paths(name, symbol)
    else:
        mapping = _normalized_dat_file_map()
        if name not in mapping:
            raise KeyError(
                f"Vendored PySCF basis snapshot does not expose basis {basisname!r}."
            )
        relpaths = (mapping[name],)

    merged: list[list[object]] = []
    for relpath in relpaths:
        if relpath.suffix == ".pydata":
            merged.extend(_python_basis(root/relpath, _std_symbol(symbol)))
            continue
        text = (root / relpath).read_text(encoding="utf-8")
        lines = _search_basis_block(text, symbol)
        merged.extend(_parse_nwchem_block(lines))
    return merged


def load_basis_from_snapshot(basisname: str, symbol: str) -> list[list[object]]:
    """Load retained named orbital basis data; return an independent mutable copy.

    Parsing availability does not imply support for ECPs or spinor integrals.
    Raw angular-momentum/kappa labels and general contraction columns survive.
    """
    return copy.deepcopy(_load_basis_from_snapshot(basisname, symbol))


__all__ = ["load_basis_from_snapshot"]
