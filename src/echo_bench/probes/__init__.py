"""ECHO-Bench probes package.

Strategy probes are controlled, instrumented INPUT policies — not synthetic
users. See :mod:`echo_bench.probes.strategy_probes` for the full guardrail and
contract.
"""

from echo_bench.probes.strategy_probes import (
    PROBES,
    StrategyProbe,
    get_probe,
)

__all__ = ["PROBES", "StrategyProbe", "get_probe"]
