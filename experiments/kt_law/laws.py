"""Law library for the IM environment.

A "law" is the stochastic activation mechanism of the diffusion. The repo's Env
hard-codes one law (independent cascade with p = 1/in_degree). Here each law is a
different mechanism, so T1 can ask whether the *decision* depends on which law is
true.

Common form.  For an inactive node v with
    a_v = number of active in-neighbours this round,
    E_v = influence accumulator (a_v + rho * E_v from the previous round),
the law gives an activation probability p_v = f(E_v, a_v, indeg_v); the node then
activates with that probability, independently of the other nodes (one round of
propagation, exactly like IM/Env.step).

  WC     weighted cascade      p = 1 - (1 - 1/indeg)^a          (the repo default)
  UNI    uniform               p = 1 - (1 - 0.05)^a             (degree-blind)
  LT     linear threshold      p = min(1, a / indeg)            (ramp, saturating)
  CC2    complex contagion     p = 1[a >= 2]                    (hard gate)
  CC3    complex contagion     p = 1[a >= 3]
  REINF  cumulative exposure   p = 1 - (1 - 1/indeg)^E, rho = 1 (past exposure keeps counting)
  DECAY  decaying influence    p = 1 - (1 - 0.05)^E, rho = 0.5  (old exposure fades)
  2HOP   two-hop influence     p = 1 - (1 - 0.05)^(a + a2)      (a2 = active nodes at distance 2)
"""

import numpy as np

STATUS_INACTIVE, STATUS_ACTIVE, STATUS_REMOVED = 0, 1, 2


class Law:
    name = 'law'
    rho = 0.0        # per-round decay of the influence accumulator (0 = instantaneous)
    two_hop = False  # whether influence travels two hops per round

    def op(self, adj):
        """Influence operator: act @ op gives each node's weighted active-neighbour count."""
        return (adj + adj @ adj).tocsr() if self.two_hop else adj

    def accumulate(self, E, a):
        return a + self.rho * E if self.rho else a

    def prob(self, E, a, indeg):
        """Activation probability; E and a broadcast as (M, n), indeg is (n,)."""
        raise NotImplementedError

    def edge_weight(self, indeg):
        """Per-target edge weight used by the repo's `score` heuristic."""
        return 1.0 / np.maximum(indeg, 1)

    def __repr__(self):
        return f'<law {self.name}>'


class WeightedCascade(Law):
    name = 'WC'

    def prob(self, E, a, indeg):
        w = 1.0 / np.maximum(indeg, 1)
        return 1.0 - np.power(1.0 - w, E)


class Uniform(WeightedCascade):
    name = 'UNI'
    c = 0.05

    def prob(self, E, a, indeg):
        return 1.0 - np.power(1.0 - self.c, E)

    def edge_weight(self, indeg):
        return np.full_like(np.asarray(indeg, dtype=np.float64), self.c)


class LinearThreshold(Law):
    name = 'LT'

    def prob(self, E, a, indeg):
        return np.clip(a / np.maximum(indeg, 1), 0.0, 1.0)


class Gate(Law):
    name = 'CC2'
    gate = 2

    def prob(self, E, a, indeg):
        return (a >= self.gate).astype(np.float32)


class Gate3(Gate):
    name = 'CC3'
    gate = 3


class Reinfection(WeightedCascade):
    """Exposure accumulates across rounds (a node stays 'primed'), no spontaneous activation."""
    name = 'REINF'
    rho = 1.0

    def prob(self, E, a, indeg):
        w = 1.0 / np.maximum(indeg, 1)
        return 1.0 - np.power(1.0 - w, E)


class Decay(WeightedCascade):
    name = 'DECAY'
    rho = 0.5
    c = 0.05

    def prob(self, E, a, indeg):
        return 1.0 - np.power(1.0 - self.c, E)

    def edge_weight(self, indeg):
        return np.full_like(np.asarray(indeg, dtype=np.float64), self.c)


class TwoHop(Uniform):
    name = '2HOP'
    two_hop = True


LAWS = {law.name: law for law in [
    WeightedCascade(), Uniform(), LinearThreshold(), Gate(), Gate3(),
    Reinfection(), Decay(), TwoHop(),
]}

# Order used by the T1 sweep: mechanisms that differ in structure, not just in a
# probability constant.
T1_LAWS = ['WC', 'UNI', 'LT', 'CC2', 'CC3', 'REINF', 'DECAY', '2HOP']


def get_law(name):
    if name not in LAWS:
        raise KeyError(f'unknown law {name!r}; choose from {sorted(LAWS)}')
    return LAWS[name]
