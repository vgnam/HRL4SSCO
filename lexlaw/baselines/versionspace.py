"""Text-free version spaces (reference probe_supply2): supplier classes given, or nothing given (B2)."""
import numpy as np

from ..model.schema import SlotSpace, SlotSpaces


def grades_spaces(net, kind="grades"):
    """kind="grades": supplier classes are observed, each slot's required class is hidden; hypotheses
    are the classes present among the links, prior = share of links. kind="none": a placeholder space
    (one all-links mask per slot) to be replaced by the blind space."""
    slots, by_consumer = [], {}
    for v in sorted(net.slots):
        for j, s in enumerate(net.slots_of[v]):
            links = net.links_of_slot(s)
            if kind == "grades":
                g = net.grade[links]
                codes = sorted(set(int(x) for x in g))
                masks = [g == c for c in codes]
                prior = np.array([float((g == c).mean()) for c in codes])
            else:
                masks, prior = [np.ones(len(links), bool)], np.array([1.0])
            slots.append(SlotSpace(v, j, s, links, None, masks, prior, None, 0, None))
            by_consumer.setdefault(v, []).append(s)
    return SlotSpaces(slots, by_consumer, {}, {})
