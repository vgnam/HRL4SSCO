"""STRATA: structure-learning for sequential social-commerce optimisation.

The package is deliberately split into a domain-independent core and two
benchmark domains:

* ``strata.core`` -- interface contract (``types.py``) and RNG helpers;
* ``strata.domains.im`` -- the influence-maximisation benchmark (IM/Env.py);
* ``strata.domains.rpp`` -- the route/profit-planning benchmark (RPP/Env.py).

Later milestones add ``laws`` (L1), ``abstraction`` (L2), ``planning`` (L3),
``diagnosis``, ``agent``, ``baselines`` and ``eval``.  M0 only installs the
foundation: types, configuration, graph/point loading, environments, hidden
laws and behaviour-log collection.
"""

__version__ = "0.1.0"
