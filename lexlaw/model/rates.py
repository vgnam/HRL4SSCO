"""Rate estimates (method Section 3.5): rho_hat = (1 + successes) / (2 + attempts),
phi_hat = (1 + failures) / (2 + producer-days at risk)."""
import numpy as np


class Rates:
    def __init__(self):
        self.succ = self.att = self.fail = self.risk = 0

    def update(self, net, up_before, repairs, up_after, shock_between=False):
        """One day: up_before on day t, repairs A_t, up_after on day t + 1. A producer-day is at risk
        when the producer is up, not repaired, and no aftershock occurs (or is detected) at t + 1."""
        rep = np.zeros(net.n, bool)
        rep[list(repairs)] = True
        self.att += int(rep.sum())
        self.succ += int((rep & up_after).sum())
        if not shock_between:
            prod = net.layer < 3
            at_risk = prod & up_before & ~rep
            self.risk += int(at_risk.sum())
            self.fail += int((at_risk & ~up_after).sum())

    @property
    def rho(self):
        return (1 + self.succ) / (2 + self.att)

    @property
    def phi(self):
        return (1 + self.fail) / (2 + self.risk)
