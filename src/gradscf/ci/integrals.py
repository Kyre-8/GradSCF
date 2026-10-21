"""CI input adaptation; shared MO transformation lives in gradscf.ao2mo."""
from gradscf.ao2mo import validate_integrals
from gradscf._memory import resolve_memory_mb
from ..scf.reference import reference_from_source as prepare_reference
from .types import CIReference


def reference_from_source(source, *, max_memory=None):
    max_memory = resolve_memory_mb(max_memory, source)
    if isinstance(source, CIReference):
        h,g = validate_integrals(source.h1,source.eri)
        if not isinstance(source.nocc,int) or not 0 <= source.nocc <= h.shape[0]:
            raise ValueError("Invalid restricted nocc")
        return CIReference(h,g,source.nocc,source.nuclear_repulsion,source.mo_energy)
    ref = prepare_reference(source, max_memory=max_memory)
    return CIReference(ref.h1,ref.eri,ref.nocc,ref.nuclear_repulsion,ref.mo_energy)
