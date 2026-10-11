"""Algorithm LexLaw on one network, one chain of E episodes (method Section 6)."""
from .agent import run_episode
from .model.belief import Belief, flag_keys
from .model.rates import Rates


def run_chain(net, schema, sup_text, req_text, seeds, K=20, cap=4, T=15, quorum=False, budget_factory=None,
              aftershock=None, phi=0.01, chat=None, nets_texts=None, reader_answers=None, log=print):
    """Returns per-episode returns and a trace (eta, flagged keys, revisions, escaped consumers).

    budget_factory: None (asap) or a callable net -> BudgetPlanner (Section 5.2).
    chat: a language-model client for revisions (Section 4.3); None disables revision.
    """
    belief = Belief(net, schema.spaces(net, sup_text, req_text), 0.2, quorum=quorum)
    rates = Rates()
    returns, trace = [], []
    for e, seed in enumerate(seeds):
        bp = budget_factory(net) if budget_factory else None
        r = run_episode(net, belief, seed, K, cap, T, em=True, budget=bp, rates=rates, phi=phi, aftershock=aftershock)
        returns.append(r)
        stats = belief.key_statistics()
        flagged = flag_keys(stats)
        revised = []
        if chat is not None and flagged:
            from .llm.synthesize import revise
            key = flagged[0]                                   # at most one revision call per episode
            schema, belief, ok = revise(chat, schema, belief, key, stats, net, sup_text, req_text,
                                        nets_texts or [(net, sup_text, req_text)], reader_answers, log)
            revised.append((key, ok))
        trace.append(dict(episode=e, ret=r, eta=belief.eta, flagged=flagged, revised=revised,
                          escaped=len(belief.escaped), schedule=bp.choices if bp else None))
    return returns, trace, schema
