"""Check the closed zero-gradient oracle against 80-digit Maple evaluation."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from test_wpbe_small_gradient import _wpbe_small_gradient_coefficient


def _maple_terms_2_to_5(w, s):
    """Direct transcription of Libxc 6.0.0 terms 2--5, without factoring B.

    https://gitlab.com/libxc/libxc/-/raw/6.0.0/maple/gga_exc/gga_x_wpbeh.mpl
    term1 starts at s⁴ log(s), so it contributes no coefficient of s².
    These small-s probes use the source's polynomial EG branch (s < .08).
    """
    b, c, d, e = map(Decimal, ("-.37170836", "-.077215461", ".57786348", "-.051955731"))
    square = s*s
    hole = ((Decimal(".00979681")*square+Decimal(".0410834")*square**2)
            /(1+Decimal(".187440")*square**2+Decimal(".00120824")*s**5
              +Decimal(".0347188")*s**6))
    f = Decimal("6.4753871")*hole+Decimal(".47965830")
    eg = Decimal("-.02628417880")-Decimal(".07117647788")*square+Decimal(".08534541323")*square**2
    aux1 = d+square*hole
    aux3 = aux1+w*w
    five_halves = aux3**2*aux3.sqrt()
    term2 = (aux1**2*b+aux1*c+2*e+aux1*square*c*f+2*square*eg)/(2*aux1**3)
    term3 = -w*(4*aux3**2*b+6*aux3*c+15*e+6*aux3*square*c*f+15*square*eg)/(8*aux1*five_halves)
    term4 = -w**3*(aux3*c+5*e+aux3*square*c*f+5*square*eg)/(2*aux1**2*five_halves)
    term5 = -w**5*(e+square*eg)/(aux1**3*five_halves)
    return -Decimal(8)/9*(term2+term3+term4+term5)


@pytest.mark.parametrize("screening", ["0", ".03", ".4", "2", "20"])
def test_closed_wpbe_s_squared_coefficient_matches_high_precision_maple(screening, record_property):
    with localcontext() as context:
        context.prec = 80
        w = Decimal(screening)
        center = _maple_terms_2_to_5(w, Decimal(0))
        coefficients = []
        for text in ("1e-12", "1e-10"):
            step = Decimal(text)
            plus, minus = _maple_terms_2_to_5(w, step), _maple_terms_2_to_5(w, -step)
            # F(s)=F(0)+B*s²+...; thus B=F''(0)/2.
            coefficient = (plus-2*center+minus)/(2*step**2)
            coefficients.append(float(coefficient))
            record_property("decimal80_B_step_"+text, str(coefficient))
    np.testing.assert_allclose(coefficients[0], coefficients[1], rtol=2e-16, atol=1e-20)
    np.testing.assert_allclose(_wpbe_small_gradient_coefficient(float(screening)),
                               coefficients, rtol=2e-13, atol=1e-18)
