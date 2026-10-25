"""Deterministic RNG helpers shared by every STRATA layer.

``make_rng(*keys)`` turns an arbitrary tuple of hashable, JSON-friendly keys
into an independent ``numpy.random.Generator``.  The derivation uses BLAKE2b,
so it is stable across processes and Python's hash randomisation (unlike
``hash``).  Use it whenever a component needs its own reproducible stream
(common random numbers, operator-specific sampling, per-law simulations).
"""

import hashlib
import json
from typing import Any

import numpy as np


def derive_seed(*keys: Any) -> int:
    """Return a stable 64-bit seed derived from ``keys``."""
    payload = json.dumps(
        keys, sort_keys=True, default=repr, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def make_rng(*keys: Any) -> np.random.Generator:
    """Return a fresh deterministic ``Generator`` for ``keys``."""
    return np.random.default_rng(derive_seed(*keys))


def seed_sequence(*keys: Any) -> np.random.SeedSequence:
    """Return a ``SeedSequence`` for ``keys`` (useful for numpy spawning)."""
    return np.random.SeedSequence(derive_seed(*keys))
