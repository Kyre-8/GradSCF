"""Packed native Coulomb ERI values (s4 and s8)."""
from .ffi import evaluate_compact


def evaluate(atm, bas, env, shape, *, layout, cart, split=0):
    return evaluate_compact(atm, bas, env, shape, layout=layout, cart=cart, split=split)
