"""Documents for DSR (method Section 2.3; quorum sentence of Section 2.4).

Levels 1-3 reproduce the reference generator's draws exactly (Section 12): catalogue lines for
producers in index order, then requirement lines for consumers in index order and slots in order,
all from one generator.
"""
import json

import numpy as np

BANDS = {  # numeric categories: (property name, canonical unit, three classes [lo, hi))
    0: ("tensile strength", "MPa", [(250, 400), (400, 550), (550, 700)]),
    1: ("maximum service temperature", "deg C", [(80, 150), (150, 250), (250, 400)]),
    2: ("rated voltage", "V", [(12, 48), (48, 240), (240, 1000)]),
    3: ("purity", "%", [(95.0, 99.0), (99.0, 99.9), (99.9, 99.999)]),
    4: ("median particle size", "micron", [(1, 10), (10, 100), (100, 1000)]),
}
MATERIAL_SUP = [  # category 5: three material families, catalogue wording
    ["AISI 304 sheet", "18/8 chromium-nickel alloy stock", "A2 grade fastener stock"],
    ["S235JR plate", "ASTM A36 plate", "mild steel, 0.2 percent carbon"],
    ["6061-T6 extrusion", "EN AW-6082 bar", "AlMgSi wrought alloy"],
]
MATERIAL_REQ = [
    ["non-magnetic corrosion-resistant stainless steel", "austenitic stainless for a wet food-processing area"],
    ["weldable low-carbon structural steel", "plain unalloyed construction steel for a painted frame"],
    ["lightweight heat-treatable aluminium alloy", "aluminium profile for a weight-critical bracket"],
]
MISSING = "No specification on file."
DUAL_SENTENCE = " Dual-sourced: two qualified suppliers must be running."


def supplier_text(cat, grade, level, rng):
    if level == 1:
        return f"Category C{cat}, grade G{grade}."
    if cat == 5:
        return f"Material: {MATERIAL_SUP[grade][rng.integers(3)]}."
    name, unit, bands = BANDS[cat]
    lo, hi = bands[grade]
    x = lo + (hi - lo) * rng.uniform(0.08, 0.92)
    k = int(rng.integers(3))
    if cat == 0:
        s = [f"{x:.0f} MPa", f"{x / 1000:.3f} GPa", f"{x * 0.1450377:.1f} ksi"][k]
    elif cat == 1:
        s = [f"{x:.0f} deg C", f"{x * 9 / 5 + 32:.0f} deg F", f"{x + 273.15:.0f} K"][k]
    elif cat == 2:
        s = [f"{x:.0f} V", f"{x / 1000:.3f} kV", f"{x * 1000:.0f} mV"][k]
    elif cat == 3:
        x = lo + (hi - lo) * rng.uniform(0.15, 0.85)
        s = [f"{x:.4f} % pure", f"total impurities {(100 - x) * 1e4:.0f} ppm", f"assay {x / 100:.6f} mass fraction"][k]
        return f"Catalogue: {s}."
    else:
        s = [f"{x:.1f} micron", f"{x / 1000:.4f} mm", f"{x * 1000:.0f} nm"][k]
    return f"Catalogue: {name} {s}."


def requirement_text(cat, grade, level, rng):
    if level == 1:
        return f"Requires category C{cat}, grade G{grade}."
    if cat == 5:
        return f"Needs {MATERIAL_REQ[grade][rng.integers(2)]}; other material families are rejected."
    name, unit, bands = BANDS[cat]
    lo, hi = bands[grade]
    return (f"Needs {name} of at least {lo:g} {unit} and below {hi:g} {unit}; "
            f"values outside this class are rejected.")


def make_texts(net, level, seed=None):
    """sup[u] for producers, req[(v, j)] for slots. Default seed 5000 + network index."""
    rng = np.random.default_rng(5000 + net.index if seed is None else seed)
    sup = {int(u): supplier_text(int(net.cat[u]), int(net.grade[u]), level, rng)
           for u in range(net.n) if net.grade[u] >= 0}
    req = {}
    for v in sorted(net.slots):
        for j, (links, m) in enumerate(net.slots[v]):
            c = int(net.cat[links[0]])
            g = int(net.grade[links[m][0]])
            if level == 3:
                r = rng.random()
                if r < 0.25:
                    req[(v, j)] = MISSING
                    continue
                if r < 0.40:
                    g = int((g + rng.integers(1, 3)) % 3)
            req[(v, j)] = requirement_text(c, g, 2 if level == 3 else level, rng)
    return sup, req


def add_quorum_sentence(net, req, level, seed=None, d_drop=0.15, d_add=0.05):
    """DSR-Q documents [specified, untested]. A line of a slot with q = 2 ends with the dual-sourcing
    sentence; at level 3 it is dropped w.p. d_drop (q = 2) and added w.p. d_add (q = 1). A missing
    line carries no sentence. Own stream default_rng(7000 + index) (not fixed by the method)."""
    assert net.quorum is not None
    rng = np.random.default_rng(7000 + net.index if seed is None else seed)
    out = {}
    for v in sorted(net.slots):
        for j in range(len(net.slots[v])):
            line = req[(v, j)]
            if line == MISSING:
                out[(v, j)] = line
                continue
            dual = net.quorum[v][j] == 2
            if level == 3:
                u = rng.random()
                dual = (u >= d_drop) if dual else (u < d_add)
            out[(v, j)] = line + (DUAL_SENTENCE if dual else "")
    return out


def load_level4(path):
    """Level-4 text [specified, untested]: a JSON file {"sup": {u: line}, "req": {"v:j": line}}."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sup = {int(k): s for k, s in d["sup"].items()}
    req = {tuple(int(x) for x in k.split(":")): s for k, s in d["req"].items()}
    return sup, req
