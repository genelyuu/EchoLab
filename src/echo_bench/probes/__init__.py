"""ECHO-Bench probes package.

Strategy probes are controlled, instrumented INPUT policies — not synthetic
users. See :mod:`echo_bench.probes.strategy_probes` for the full guardrail and
contract.
"""

from echo_bench.probes.probe_overlap import (
    PROBE_OVERLAP_THRESHOLD,
    probe_overlap_audit,
)
from echo_bench.probes.strategy_probes import (
    DEFAULT_PROBE_SET,
    PROBES,
    StrategyProbe,
    get_probe,
)

__all__ = [
    "DEFAULT_PROBE_SET",
    "PROBES",
    "PROBE_OVERLAP_THRESHOLD",
    "StrategyProbe",
    "get_probe",
    "probe_overlap_audit",
]
