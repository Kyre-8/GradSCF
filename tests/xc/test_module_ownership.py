"""All consumers share the XC package rather than a second implementation."""
import ast
from pathlib import Path


def test_consumers_do_not_import_the_retired_xc_implementation():
    source = Path(__file__).resolve().parents[2] / 'src'
    old = source / 'gradscf/dft/libxc_jax'
    imports = []
    for path in source.rglob('*.py'):
        if old in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and 'libxc_jax' in (node.module or ''):
                imports.append(str(path.relative_to(source)))
            if isinstance(node, ast.Import):
                imports.extend(str(path.relative_to(source)) for alias in node.names
                               if 'libxc_jax' in alias.name)
    assert not imports, imports


def test_only_one_classical_xc_implementation_is_installed():
    root = Path(__file__).resolve().parents[2]
    assert (root / 'src/gradscf/dft/xc/__init__.py').is_file()
    assert not (root / 'src/gradscf/dft/libxc_jax').exists()
