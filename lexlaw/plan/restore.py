"""Low level: restoration-set greedy (method Section 5.1; reference plan_model / plan_q)."""
from ..envs.dsr import operating


def restore_greedy(net, up, compat, room, quorum=None) -> list:
    """Repairs for today under the model (compat, quorum), at most `room` of them.

    Repeatedly adds the restoration set of the customer with the best weight / |set|; a slot is
    restored by the q smallest restoration sets of its believed-compatible links (earlier link first
    on ties); per-slot choices are united ("tree approximation"); planned repairs are assumed to succeed.
    """
    up = up.copy()
    repairs = []
    starts, lens, lnode = net.slot_start, net.slot_len, net.link_node
    while room > 0:
        op_m = operating(net, up, compat, quorum)
        memo = {}

        def restore(v):
            if v in memo:
                return memo[v]
            if op_m[v]:
                memo[v] = frozenset()
                return memo[v]
            need = set() if up[v] else {v}
            for s in net.slots_of.get(v, ()):
                a = starts[s]
                q = 1 if quorum is None else int(quorum[s])
                sets = []
                for e in range(a, a + lens[s]):
                    if compat[e]:
                        r = restore(int(lnode[e]))
                        if r is not None:
                            sets.append(r)
                if len(sets) < q:
                    memo[v] = None
                    return None
                sets.sort(key=len)                    # stable: earlier link first on ties
                for r in sets[:q]:
                    need |= r
            memo[v] = frozenset(need)
            return memo[v]

        best, best_val = None, 0.0
        for c in net.idx[3]:
            c = int(c)
            if op_m[c]:
                continue
            s = restore(c)
            if s is None or len(s) == 0 or len(s) > room:
                continue
            val = net.w[c] / len(s)
            if val > best_val:
                best, best_val = s, val
        if best is None:
            break
        for u in best:
            up[u] = True
            repairs.append(int(u))
        room -= len(best)
    return repairs
