"""L1 law layer of STRATA.

The modules in this package turn logged transitions into a finite hypothesis
set of hidden laws:

``features``   per-round sufficient statistics ``(a, E, a2, eligible, ...)``
``dsl``        the law DSL (Bernoulli effects for IM, continuous drift
               distributions for RPP) and the OneLife product law
``enumerate``  systematic structure enumeration by level
``fit``        maximum-likelihood fitting of structures to logs

Later layers (``laws.hypotheses``) consume the fitted laws produced here.
"""

from strata.laws import dsl, enumerate, features, fit

__all__ = ["features", "dsl", "enumerate", "fit"]
