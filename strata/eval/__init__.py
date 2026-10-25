"""Benchmark protocol and command-line entry points of STRATA.

``protocol``  instances, behaviour logs, closed-loop evaluation and regret;
``run``       ``python -m strata.eval.run`` writes one CSV row per
              (instance, method, episode);
``report``    ``python -m strata.eval.report`` prints mean/max regret tables.

The baseline policies themselves live in :mod:`strata.baselines`; the STRATA
agent from later milestones is wrapped as a policy with the same
``reset/act/observe`` interface defined in :mod:`strata.eval.protocol`.
"""
