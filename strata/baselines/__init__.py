"""Baselines for the STRATA benchmark protocol.

``flat``         flat-<op>-<alloc> policies and the ``best_flat`` selector;
``model_based``  WC-default, LibA, LibB, LibC and the oracle-law reference;
``ws_option``    adapter for the original WS-Option agents (IM/Agent.py,
                 RPP/Agent.py), which requires PyTorch.
"""

from strata.baselines.flat import FlatPolicy, allocation_k, best_flat
from strata.baselines.model_based import (
    EmpiricalDeltas,
    LibA,
    LibB,
    LibC,
    Oracle,
    WCDefault,
)

__all__ = [
    "FlatPolicy",
    "allocation_k",
    "best_flat",
    "EmpiricalDeltas",
    "LibA",
    "LibB",
    "LibC",
    "Oracle",
    "WCDefault",
]
