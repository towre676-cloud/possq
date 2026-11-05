"""Monotone reuse of exact raw-response certificates as measurements improve.

Additional readings cannot enlarge the compatible model set. An optimizer can
still return a numerically worse polynomial. Retaining an old certificate,
embedding its raw coefficients in the extended basis, and replaying the exact
verifier prevents this numerical regression without an optimality assumption.
"""
from fractions import Fraction as F

from .resolvent import ResolventCertificate, verify_resolvent_certificate


def extend_certificate(old_data, certificate, new_data):
    """Lift a certificate when old error intervals contain the new intervals.

    New shifts may be added, existing readings may become more precise, and
    repeated shifts are allowed. Support and target conventions stay fixed.
    All input containments and final side conditions are checked exactly.
    """
    if not verify_resolvent_certificate(old_data, certificate):
        raise ValueError('old certificate does not verify')
    if old_data.I != new_data.I:
        raise ValueError('support intervals must agree')
    if abs(new_data.mass-old_data.mass)+new_data.mass_eps > old_data.mass_eps:
        raise ValueError('new mass interval does not refine the old interval')
    coeffs = [F(0)] * len(new_data.values)
    coeffs[0] = certificate.coeffs[0]
    for k, shift in enumerate(old_data.shifts):
        candidates = [j for j, s in enumerate(new_data.shifts) if s == shift and
                      abs(new_data.hhat[j]-old_data.hhat[k])+new_data.eps[j] <= old_data.eps[k]]
        if not candidates:
            raise ValueError('an old reading was removed or its error interval widened')
        coeffs[candidates[0]+1] += certificate.coeffs[k+1]
    value = new_data.upper(coeffs) if certificate.side == 'upper' else new_data.lower(coeffs)
    result = ResolventCertificate(certificate.side, value, coeffs, certificate.shift,
                                  certificate.S.copy(), certificate.lp_value)
    if not verify_resolvent_certificate(new_data, result):
        raise ArithmeticError('lifted certificate failed exact verification')
    if certificate.side == 'upper' and result.value > certificate.value:
        raise ArithmeticError('refined upper interval worsened the certificate')
    if certificate.side == 'lower' and result.value < certificate.value:
        raise ArithmeticError('refined lower interval worsened the certificate')
    return result


def best_certificate(data, candidates):
    """Select the strongest independently verified certificate on one target."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError('at least one certificate is required')
    if any(not verify_resolvent_certificate(data,c) for c in candidates):
        raise ValueError('candidate does not verify')
    side, target = candidates[0].side, candidates[0].S
    if any(c.side != side or c.S != target for c in candidates):
        raise ValueError('certificates must share a target and side')
    return (min if side == 'upper' else max)(candidates,key=lambda c:c.value)
