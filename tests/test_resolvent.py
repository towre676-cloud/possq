"""Soundness and falsification controls for raw-response certificates."""
from dataclasses import replace
from fractions import Fraction as F

import pytest

from possq.resolvent import (
    ResolventCertificate, ResolventData, ResolventWitness,
    certify_resolvent_candidate, resolvent_mass_certificate,
    resolvent_primal_witness, verify_resolvent_certificate,
    verify_resolvent_witness,
)


def fixture(noise=F(1, 10**5)):
    atoms = [(F(1, 4), F(1, 3)), (F(3, 4), F(2, 3))]
    shifts = [F(1, 8), F(1, 4), F(1, 2), F(1), F(2)]
    h = [sum((w / (x + s) for x, w in atoms), F(0)) for s in shifts]
    data = ResolventData((F(1, 8), F(1)), shifts, h, [noise] * len(shifts))
    return data, atoms, [(F(1, 8), F(1, 2))]


def test_positive_support_and_no_denominator_poles():
    with pytest.raises(ValueError, match="positive support"):
        ResolventData((0, 1), [1], [1], [0])
    with pytest.raises(ValueError, match="positive"):
        ResolventData((F(1, 8), 1), [-F(1, 8)], [1], [0])
    with pytest.raises(ValueError, match="positive"):
        ResolventData((F(1, 8), 1), [-F(1, 4)], [1], [0])
    # Negative shifts are legitimate when the pole remains strictly outside I.
    d = ResolventData((F(1, 4), 1), [-F(1, 8)], [1], [0])
    assert d.phi(F(1, 4)) == [1, 8]


def test_raw_mass_bounds_and_exact_atomic_brackets():
    data, atoms, S = fixture()
    truth = F(1, 3)
    upper = resolvent_mass_certificate(data, S, "upper")
    lower = resolvent_mass_certificate(data, S, "lower")
    assert lower.value <= truth <= upper.value
    assert upper.value - lower.value < F(3, 50)
    for cert in (upper, lower):
        assert verify_resolvent_certificate(data, cert)
        assert verify_resolvent_certificate(data, cert.as_dict())
        rebuilt = ResolventCertificate.from_dict(cert.as_dict())
        assert rebuilt.coeffs == cert.coeffs and rebuilt.value == cert.value
    wu = resolvent_primal_witness(data, S, "upper", extra_points=[x for x, _ in atoms])
    wl = resolvent_primal_witness(data, S, "lower", extra_points=[x for x, _ in atoms])
    assert wu is not None and wl is not None
    assert verify_resolvent_witness(data, wu.as_dict())
    assert verify_resolvent_witness(data, wl)
    assert 0 <= upper.value - wu.value < F(1, 1000)
    assert 0 <= wl.value - lower.value < F(1, 1000)


def test_bad_proposal_is_repaired_and_tampering_is_rejected():
    data, _, S = fixture()
    bad = [F(-2)] + [F(0)] * len(data.shifts)
    # This impossible upper sandwich is repaired by at least three units.
    cert = certify_resolvent_candidate(data, S, bad, "upper")
    assert cert.shift >= 3
    assert verify_resolvent_certificate(data, cert)
    assert not verify_resolvent_certificate(data, replace(cert, coeffs=bad))
    assert not verify_resolvent_certificate(data, replace(cert, value=cert.value - F(1, 100)))
    assert not verify_resolvent_certificate(data, replace(cert, shift=F(-1)))
    # The denominator check is part of the independent verification boundary.
    data.shifts[0] = -data.I[0]
    assert not verify_resolvent_certificate(data, cert)


def test_exact_known_law_and_witness_falsification_controls():
    data, atoms, S = fixture(F(0))
    wit = ResolventWitness("upper", F(1, 3), atoms, S)
    assert verify_resolvent_witness(data, wit)
    assert verify_resolvent_witness(data, ResolventWitness.from_dict(wit.as_dict()))
    assert not verify_resolvent_witness(data, replace(wit, value=F(1, 2)))
    assert not verify_resolvent_witness(data, replace(wit, atoms=[(F(1, 4), F(1))]))
    assert not verify_resolvent_witness(data, replace(wit, atoms=[(F(-1), F(1))]))
    # Raw-response optimization may approximate, but every returned bound is exact.
    for side in ("upper", "lower"):
        cert = resolvent_mass_certificate(data, S, side)
        assert verify_resolvent_certificate(data, cert)
        assert cert.value >= F(1, 3) if side == "upper" else cert.value <= F(1, 3)


def test_empty_full_and_multiple_interval_targets():
    data, _, _ = fixture()
    for side in ("upper", "lower"):
        empty = resolvent_mass_certificate(data, [], side)
        full = resolvent_mass_certificate(data, [data.I], side)
        assert empty.value == 0 and full.value == 1
        assert verify_resolvent_certificate(data, empty)
        assert verify_resolvent_certificate(data, full)
    # Both true atoms are included; clipping and merging retain the closed convention.
    S = [(F(-1), F(1, 3)), (F(2, 3), F(2))]
    cert = resolvent_mass_certificate(data, S, "upper")
    assert cert.value >= 1 and verify_resolvent_certificate(data, cert)


def test_large_physical_scale_uses_raw_response_objective():
    atoms = [(F(100), F(2, 5)), (F(1000), F(3, 5))]
    shifts = [F(1), F(10), F(100), F(1000)]
    h = [sum(w / (x + s) for x, w in atoms) for s in shifts]
    data = ResolventData((F(10), F(2000)), shifts, h, [F(1, 10**8)] * len(shifts))
    cert = resolvent_mass_certificate(data, [(F(10), F(300))], "upper")
    assert cert.value >= F(2, 5)
    assert cert.value == data.upper(cert.coeffs)
    assert verify_resolvent_certificate(ResolventData.from_dict(data.as_dict()), cert)


def test_singleton_targets_and_mutated_budgets_cannot_claim_zero_atom_mass():
    data, atoms, _ = fixture()
    S = [(F(1, 4), F(1, 4))]
    upper = resolvent_mass_certificate(data, S)
    assert upper.value >= F(1, 3)
    assert upper.S == S
    assert verify_resolvent_certificate(data, upper)
    witness = ResolventWitness("upper", F(1, 3), atoms, S)
    assert verify_resolvent_witness(data, witness)
    data.eps[0] = F(-1)
    assert not verify_resolvent_certificate(data, upper)
    assert not verify_resolvent_witness(data, witness)


def test_twelve_raw_readings_resolve_narrow_extrema_without_large_repair():
    # High-order cancellation used to make binary64 derivative roots miss a
    # narrow trough and inflate the repair by 1e-2. Arb root proposals plus
    # independent Sturm verification retain the useful twelve-reading result.
    axes = [F(1, 32), F(1, 16), F(1, 8), F(1, 4), F(1, 2)]
    xs = [F(1, 64) + 2 * sum((w for j, w in enumerate(axes) if mask >> j & 1), F(0))
          for mask in range(32)]
    shifts = [F(1, 16), F(1, 512), F(1, 32), F(1), F(1, 8), F(2),
              F(1, 256), F(1, 128), F(1, 4), F(1, 2), F(4), F(8)]
    h0 = sum((1 / x for x in xs), F(0)) / 32
    readings = [sum((1 / (x + s) for x in xs), F(0)) / 32 for s in shifts]
    data = ResolventData((F(1, 64), F(125, 64)), shifts, readings,
                         [h0 / 10**10] * len(shifts))
    S = [(F(1, 64), F(1, 8))]
    upper = resolvent_mass_certificate(data, S, "upper")
    lower = resolvent_mass_certificate(data, S, "lower")
    assert lower.value <= F(1, 16) <= upper.value
    assert upper.value < F(83, 1000) and lower.value > F(61, 1000)
    assert upper.shift < F(1, 10**5) and lower.shift < F(1, 10**5)
    assert verify_resolvent_certificate(data, upper)
    assert verify_resolvent_certificate(data, lower)
