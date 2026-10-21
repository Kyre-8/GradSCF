"""One SCF budget plans AO blocks without changing the quadrature."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gradscf.integrals.grids import blocks
from gradscf.scf.inputs import grid
from gradscf.scf.inputs import assembly


def _basis(nao=64, nprimitive=16):
    return SimpleNamespace(
        nao=nao,
        ao_centers=np.zeros((nao, 3)),
        ao_exponents_padded=np.ones((nao, nprimitive)),
        ao_coefficients_padded=np.ones((nao, nprimitive)),
        ao_nprims_tuple=(nprimitive,) * nao,
        ao_angulars=((0, 0, 0),) * nao,
    )


def _provider(basis, *, max_memory, grid_max_memory=None, requested=0,
              ngrids=100_000, scf_history_matrices=2, fixed_workspace_bytes=0):
    return grid._streamed_grid(
        basis, np.zeros((ngrids, 3)), np.ones(ngrids),
        needs_ao_laplacian=False, grid_block_size=requested,
        grid_max_memory=grid_max_memory, max_memory=max_memory,
        scf_history_matrices=scf_history_matrices,
        fixed_workspace_bytes=fixed_workspace_bytes,
    )


def test_more_memory_increases_the_automatic_ao_block():
    basis = _basis()
    small = _provider(basis, max_memory=128.)
    large = _provider(basis, max_memory=4096.)
    assert 1 <= small.block_size < large.block_size <= 16384
    assert small.adaptive and large.adaptive
    assert large.ngrids == small.ngrids == 100_000


def test_explicit_grid_cap_cannot_spend_more_than_the_total_budget():
    basis = _basis()
    automatic = _provider(basis, max_memory=128.)
    large_cap = _provider(basis, max_memory=128., grid_max_memory=60000.)
    small_cap = _provider(basis, max_memory=4096., grid_max_memory=16.)
    assert large_cap.block_size == automatic.block_size
    assert small_cap.block_size < _provider(basis, max_memory=4096.).block_size
    assert not large_cap.adaptive and not small_cap.adaptive


def test_unrestricted_history_reserve_reduces_available_ao_workspace():
    basis = _basis(nao=128)
    restricted = _provider(basis, max_memory=128., scf_history_matrices=2)
    unrestricted = _provider(basis, max_memory=128., scf_history_matrices=6)
    assert 1 <= unrestricted.block_size < restricted.block_size


@pytest.mark.parametrize("requested", [3, 256])
def test_explicit_block_size_remains_an_upper_bound(requested):
    provider = _provider(_basis(), max_memory=4096., requested=requested)
    assert provider.block_size == requested
    assert not provider.adaptive


def test_dense_auto_planning_counts_primitive_workspace():
    # The resident AO+gradient payload fits comfortably in 1 GB, but evaluating
    # all primitive channels at once would need much more than that budget.
    basis = _basis(nao=32, nprimitive=64)
    provider = _provider(basis, max_memory=1024., ngrids=100_000)
    assert provider is not None
    assert provider.block_size < provider.ngrids
    assert _provider(basis, max_memory=1024., requested=-1, ngrids=100) is None
    assert _provider(basis, max_memory=1024., ngrids=100) is None


def test_ao_point_estimate_accepts_basis_and_provider_without_array_copies():
    basis = _basis()
    provider = blocks.AOGrid.from_basis(
        basis, np.zeros((9, 3)), np.ones(9), block_size=3, adaptive=True,
    )
    for deriv in (0, 1, 2):
        assert blocks.ao_point_bytes(basis, deriv=deriv) == blocks.ao_point_bytes(
            provider, deriv=deriv,
        )
    assert blocks.ao_point_bytes(basis, deriv=2) > blocks.ao_point_bytes(basis, deriv=0)
    leaves, structure = jax.tree_util.tree_flatten(provider)
    assert len(leaves) == 5
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert restored.adaptive and restored.block_size == 3


def test_cpu_automatic_block_uses_the_measured_performance_cap():
    basis = _basis()
    assert blocks.ao_block_size(basis, deriv=1, max_memory=4096.) == 4096


@pytest.mark.parametrize("requested", [8192, 16384])
def test_explicit_large_ao_block_is_permitted_within_the_budget(requested):
    assert blocks.ao_block_size(_basis(), deriv=1, max_memory=4096.,
                                requested=requested) == requested


def test_gpu_automatic_block_uses_a_larger_cap_without_device_enumeration(monkeypatch):
    def reject_device_enumeration(*args, **kwargs):
        raise AssertionError("Planning must inspect only the supplied device.")

    monkeypatch.setattr(jax, "devices", reject_device_enumeration)
    device = SimpleNamespace(platform="gpu")
    assert blocks.ao_block_size(_basis(), deriv=1, max_memory=4096.,
                                device=device) == 16384


def test_known_scf_workspace_is_charged_before_ao_planning():
    basis = _basis()
    free = _provider(basis, max_memory=128.)
    charged = _provider(basis, max_memory=128., fixed_workspace_bytes=64_000_000)
    assert 1 <= charged.block_size < free.block_size
    with pytest.raises(ValueError, match="memory|Memory"):
        _provider(basis, max_memory=128., fixed_workspace_bytes=128_000_000)


def test_traced_grid_planning_does_not_enumerate_devices(monkeypatch):
    def reject_device_enumeration(*args, **kwargs):
        raise AssertionError("Grid planning must not initialize unrelated devices.")

    monkeypatch.setattr(jax, "devices", reject_device_enumeration)
    coords = SimpleNamespace(shape=(3, 3), dtype=np.dtype("float64"))
    assert grid._streamed_grid(_basis(), coords, np.ones(3),
        needs_ao_laplacian=False, grid_block_size=-1, grid_max_memory=None,
        max_memory=128., scf_max_cycle=0) is None


def test_grid_planning_resolves_a_legacy_callable_device(monkeypatch):
    expected = SimpleNamespace(platform="cpu")
    coords = SimpleNamespace(shape=(3, 3), dtype=np.dtype("float64"),
                             device=lambda: expected)

    def workspace(max_memory, *, resident_bytes, device):
        assert device is expected
        return 100_000_000

    monkeypatch.setattr(grid, "workspace_bytes", workspace)
    assert grid._streamed_grid(_basis(), coords, np.ones(3),
        needs_ao_laplacian=False, grid_block_size=-1, grid_max_memory=None,
        max_memory=128., scf_max_cycle=0) is None


def test_grid_cache_key_tracks_total_and_optional_grid_budgets(monkeypatch):
    monkeypatch.setattr(grid, "_active_device_cache_key", lambda: ("cpu", 0))
    spec = SimpleNamespace(symbols=("H",), coords_bohr=np.zeros((1, 3)))
    options = dict(spec=spec, basis="sto-3g", max_l=3, grids_level=3,
                   precompute_eri_groups=False, needs_ao_laplacian=False,
                   grid_max_memory=None)
    small = grid._grid_ao_input_cache_key(**options, max_memory=128.)
    large = grid._grid_ao_input_cache_key(**options, max_memory=60000.)
    capped = grid._grid_ao_input_cache_key(
        **(options | {"grid_max_memory": 256.}), max_memory=60000.,
    )
    assert small != large != capped
    assert small != grid._grid_ao_input_cache_key(
        **options, max_memory=128., scf_history_matrices=6,
    )
    assert small != grid._grid_ao_input_cache_key(
        **options, max_memory=128., fixed_workspace_bytes=64_000_000,
    )


def test_grid_cache_hit_replans_blocks_without_rebuilding_grid_arrays(monkeypatch):
    monkeypatch.setattr(grid, "_GRID_AO_INPUT_CACHE", {})
    monkeypatch.setattr(grid, "_active_device_cache_key", lambda: ("cpu", 0))
    capacity = [10_000_000]
    monkeypatch.setattr(grid, "workspace_bytes", lambda *args, **kwargs: capacity[0])
    basis, coords, weights = _basis(), np.zeros((100_000, 3)), np.ones(100_000)
    spec = SimpleNamespace(symbols=("H",), coords_bohr=np.zeros((1, 3)))

    def reject_dense_ao(*args, **kwargs):
        raise AssertionError("Replanning a streamed cache must not evaluate full-grid AO.")

    options = dict(basis_from_molecule_spec=lambda *args, **kwargs: basis,
        build_molecular_grid_from_spec=lambda *args, **kwargs: (coords, weights),
        evaluate_cartesian_ao_with_derivatives=reject_dense_ao,
        spec=spec, basis="sto-3g", max_l=3, grids_level=3,
        precompute_eri_groups=False, needs_ao_laplacian=False, max_memory=128.)
    first = grid._cached_grid_ao_input_bundle(**options)
    assert grid._cached_grid_ao_input_bundle(**options) is first
    capacity[0] = 5_000_000
    smaller = grid._cached_grid_ao_input_bundle(**options)
    assert smaller.grid_ao.block_size < first.grid_ao.block_size
    assert smaller.coords is first.coords and smaller.grid_weights is first.grid_weights
    assert smaller.ao is first.ao and smaller.ao_deriv1 is first.ao_deriv1
    assert grid._cached_grid_ao_input_bundle(**options) is smaller


@pytest.mark.parametrize("xc_spec", ["pbe", "cam-b3lyp"])
def test_full_eri_allocation_rejected_before_exceeding_the_total_budget(monkeypatch, xc_spec):
    from gradscf.scf.rks import RKSConfig

    context = SimpleNamespace(geometry_is_traced=False, basis=_basis(nao=200),
                              coords=np.zeros((0, 3)), grid_weights=np.zeros(0),
                              grid_ao_bundle=None, grid_ao=None)
    monkeypatch.setattr(assembly, "_prepare_basis_grid_context", lambda **kwargs: context)
    monkeypatch.setattr(assembly, "prepare_basis", lambda *args, **kwargs: (
        SimpleNamespace(nao=200), None,
    ))

    class Plan:
        def evaluate(self, operator, parameters, **kwargs):
            assert operator != "eri", "Budget check must run before an ERI allocation."
            return np.zeros((200, 200))

    monkeypatch.setattr(assembly, "make_plan", lambda *args, **kwargs: Plan())
    with pytest.raises(ValueError, match="(?i)memory.*direct|budget.*direct"):
        assembly.build_rks_integral_inputs(
            atom="H 0 0 0; H 0 0 1", basis="sto-3g", xc_spec=xc_spec,
            config=RKSConfig(xc_spec=xc_spec, max_memory=128., jk_backend="full"),
        )


def test_eri_preflight_includes_the_planned_ao_workspace():
    from gradscf.scf.rks import RKSConfig

    basis = _basis(nao=40)
    coords, weights = np.zeros((1000, 3)), np.ones(1000)
    provider = blocks.AOGrid.from_basis(basis, coords, weights, block_size=128)
    context = SimpleNamespace(basis=basis, coords=coords, grid_weights=weights,
                              grid_ao=provider, grid_ao_bundle=None,
                              geometry_is_traced=False)
    # The 5.4 MB pair matrix alone fits the 16 MB total budget after reserves;
    # its combination with the planned 8.5 MB AO workspace does not.
    with pytest.raises(ValueError, match="AO workspace"):
        assembly._check_eri_memory(context, RKSConfig(max_memory=16.),
            mode="native", rsh=None, needs_ao_laplacian=False,
            scf_history_matrices=2)
