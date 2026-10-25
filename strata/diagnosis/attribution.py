"""Attribute what went wrong in an episode: law, abstraction or planner.

For every executed macro the agent stores the realised round reward and a
predictive band of that reward under the MAP hypothesis (simulated from the
same pre-round state with the same seeds / route). Then:

* ``law``          the MAP law fails calibration on the episode's transitions;
* ``abstraction``  the law is calibrated but the realised reward falls outside
                   the predictive band (the macro did not do what the planner
                   expected of it);
* ``ok``           both checks pass.

The bands are computed in :meth:`strata.agent.StrataAgent.observe` when the
agent runs with ``diagnose=True``.
"""

from __future__ import annotations

from typing import Any, Dict, List


def attribute(records: List[Dict[str, Any]], law_p_value: float, alpha: float = 0.01):
    out = []
    for rec in records:
        band = rec.get("band")
        reward = rec.get("reward")
        if law_p_value < alpha:
            label = "law"
        elif band is not None and reward is not None and not (band[0] - 1e-9 <= reward <= band[1] + 1e-9):
            label = "abstraction"
        else:
            label = "ok"
        out.append({"t": rec.get("t"), "macro": rec.get("macro"), "reward": reward,
                    "band": band, "label": label})
    summary = {lab: sum(1 for o in out if o["label"] == lab) for lab in ("law", "abstraction", "ok")}
    return out, summary
