from fractions import Fraction as F

import pytest

from possq.resolvent import ResolventData, certify_resolvent_candidate, verify_resolvent_certificate
from possq.resolvent_design import extend_certificate, best_certificate


def test_added_readings_and_narrower_budgets_cannot_weaken_retained_certificate():
    old=ResolventData((F(1),F(3)),[F(1)],[F(1,3)],[F(1,100)])
    cert=certify_resolvent_candidate(old,[(F(1),F(2))],[F(0),F(3)],'upper')
    new=ResolventData(old.I,[F(2),F(1)],[F(1,4),F(1,3)],[F(1,1000),F(1,1000)])
    lifted=extend_certificate(old,cert,new)
    assert verify_resolvent_certificate(new,lifted)
    assert lifted.value < cert.value
    constant=certify_resolvent_candidate(new,[(F(1),F(2))],[F(1),F(0),F(0)],'upper')
    assert best_certificate(new,[constant,lifted]).value <= constant.value


def test_widened_or_missing_measurement_is_rejected():
    data=ResolventData((F(1),F(3)),[F(1)],[F(1,3)],[F(1,100)])
    cert=certify_resolvent_candidate(data,[(F(1),F(2))],[F(1),F(0)],'upper')
    for new in [ResolventData(data.I,[F(1)],[F(1,3)],[F(1,10)]),
                ResolventData(data.I,[F(2)],[F(1,4)],[F(1,100)])]:
        with pytest.raises(ValueError):extend_certificate(data,cert,new)
