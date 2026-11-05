"""possq — certified spectral recovery from noisy moments via positive-square certificates."""
__version__ = "0.4.0"
from .exact import Basis, certify_positive, count_distinct_roots, complement
from .bounds import (MomentData, leakage_bound_square, separation_floor, mass_certificate,
                     primal_witness)
from .recover import (TrueMeasure, noisy_data, prony, presence_radius, cluster_enclosure,
                      hankel_spectrum, rationalize)

from .resolvent import (ResolventData, ResolventCertificate, ResolventWitness,
                        resolvent_mass_certificate, resolvent_primal_witness,
                        certify_resolvent_candidate, verify_resolvent_certificate,
                        verify_resolvent_witness)
