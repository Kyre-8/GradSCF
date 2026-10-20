"""PW vector parameters must survive upstream factory/structure conversion."""
from collections import namedtuple
from types import SimpleNamespace

import numpy as np

from gradscf.dft.xc._jax_xc import _JAXXCModule


def _module():
    Params = namedtuple("Params", "pp a alpha1 beta1 beta2 beta3 beta4 fz20")
    P = namedtuple("P", "params")
    def broken_get_p(name, polarized, *values):
        arrays = [np.repeat(values[i], 3) for i in range(0, 21, 3)]
        return P(Params(*arrays, values[21]))
    namespace = {"get_p": broken_get_p}
    exec('''def lda_c_pw(polarized=True, _a_1_=0.015545):
    p = get_p("lda_c_pw", polarized, 1.,1.,1., .031091,_a_1_,.016887,
        .2137,.20548,.11125, 7.5957,14.1189,10.357, 3.5876,6.1977,3.6231,
        1.6382,3.3662,.88026, .49294,.62517,.49671, 1.709921)
    def functional(rho, r, mo=None):
        return p.params
    return functional
''', namespace)
    return SimpleNamespace(lda_c_pw=namespace["lda_c_pw"]), broken_get_p


def test_pw_parameter_vectors_are_preserved_without_global_patching():
    raw, original = _module()
    adapted = _JAXXCModule(raw).lda_c_pw(polarized=True)(None, None)
    np.testing.assert_allclose(adapted.a, [.031091,.015545,.016887], rtol=0, atol=0)
    np.testing.assert_allclose(adapted.beta4, [.49294,.62517,.49671], rtol=0, atol=0)
    assert raw.lda_c_pw.__globals__["get_p"] is original
    np.testing.assert_allclose(raw.lda_c_pw()(None,None).a, [.031091]*3)


def test_pw_nondefault_parameters_and_unpolarized_calls_are_preserved():
    raw, _ = _module()
    factory = _JAXXCModule(raw).lda_c_pw
    for polarized in [True, False]:
        p = factory(polarized=polarized, _a_1_=.02)(None,None)
        np.testing.assert_allclose(p.a, [.031091,.02,.016887], rtol=0, atol=0)
