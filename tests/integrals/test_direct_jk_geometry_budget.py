"""Static native workspace includes the Hessian shell's nine components."""
from types import SimpleNamespace


def test_workspace_covers_nine_component_contracted_shell():
    from gradscf.integrals.backends.native.jk import native_jk_workspace_bytes

    topology = SimpleNamespace(angular_momenta=(0,), primitive_counts=(1,),
        contraction_counts=(64,), nuclear_charges=(1,), cart=True)
    provider = SimpleNamespace(plan=SimpleNamespace(topology=topology))
    # Metadata only: no AO, density or 64^4 shell array is constructed.
    required = 9*64**4*8
    actual = native_jk_workspace_bytes(provider, workers=1, cutoff=0.)
    assert actual >= required, (actual, required)
