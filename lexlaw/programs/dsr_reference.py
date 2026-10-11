"""Hand-written reference schema program for DSR text (method Sections 3.1 and 10.4, gate G1).

It is loaded by lexlaw.model.schema.load_program as source text, so it may only import re and
math and must not use dunder names. Material families are labelled "f0", "f1", "f2" so that the
canonical predicate order equals the reference's class indices.
"""
import re

MATERIAL_SUP = [
    ["aisi 304 sheet", "18/8 chromium-nickel alloy stock", "a2 grade fastener stock"],
    ["s235jr plate", "astm a36 plate", "mild steel, 0.2 percent carbon"],
    ["6061-t6 extrusion", "en aw-6082 bar", "almgsi wrought alloy"],
]
MATERIAL_REQ = [
    ["non-magnetic corrosion-resistant stainless steel", "austenitic stainless for a wet food-processing area"],
    ["weldable low-carbon structural steel", "plain unalloyed construction steel for a painted frame"],
    ["lightweight heat-treatable aluminium alloy", "aluminium profile for a weight-critical bracket"],
]
KEY_OF_NAME = {"tensile strength": "tensile", "maximum service temperature": "temp", "rated voltage": "volt",
               "purity": "purity", "median particle size": "particle"}
KSI_PER_MPA = 0.1450377


def _num(pattern, t):
    m = re.search(pattern, t)
    return float(m.group(1)) if m else None


def _value(t):
    """Catalogue value in the canonical unit of its property, or None."""
    if " gpa" in t:
        x = _num(r"([\d.]+) gpa", t)
        return None if x is None else x * 1000
    if " ksi" in t:
        x = _num(r"([\d.]+) ksi", t)
        return None if x is None else x / KSI_PER_MPA
    if " mpa" in t:
        return _num(r"([\d.]+) mpa", t)
    if "deg f" in t:
        x = _num(r"([\d.]+) deg f", t)
        return None if x is None else (x - 32) * 5 / 9
    if re.search(r"[\d.]+ k\.", t):
        x = _num(r"([\d.]+) k\.", t)
        return None if x is None else x - 273.15
    if "deg c" in t:
        return _num(r"([\d.]+) deg c", t)
    if " kv" in t:
        x = _num(r"([\d.]+) kv", t)
        return None if x is None else x * 1000
    if " mv" in t:
        x = _num(r"([\d.]+) mv", t)
        return None if x is None else x / 1000
    if re.search(r"[\d.]+ v\.", t):
        return _num(r"([\d.]+) v\.", t)
    if "ppm" in t:
        x = _num(r"([\d.]+) ppm", t)
        return None if x is None else 100 - x / 1e4
    if "mass fraction" in t:
        x = _num(r"assay ([\d.]+)", t)
        return None if x is None else x * 100
    if "% pure" in t:
        return _num(r"([\d.]+) % pure", t)
    if " nm" in t:
        x = _num(r"([\d.]+) nm", t)
        return None if x is None else x / 1000
    if " mm" in t:
        x = _num(r"([\d.]+) mm", t)
        return None if x is None else x * 1000
    if "micron" in t:
        return _num(r"([\d.]+) micron", t)
    return None


def parse_supplier(text):
    t = text.lower()
    code = re.search(r"category c(\d+), grade g(\d+)", t)
    if code:
        return {"key": "c" + code.group(1), "value": "g" + code.group(2)}
    if t.startswith("material:"):
        for g, names in enumerate(MATERIAL_SUP):
            if any(n in t for n in names):
                return {"key": "material", "value": "f" + str(g)}
        return {"key": "material", "value": None}
    val = _value(t)
    if any(x in t for x in (" gpa", " ksi", " mpa")):
        key = "tensile"
    elif "deg f" in t or "deg c" in t or re.search(r"[\d.]+ k\.", t):
        key = "temp"
    elif " kv" in t or " mv" in t or re.search(r"[\d.]+ v\.", t):
        key = "volt"
    elif "ppm" in t or "mass fraction" in t or "% pure" in t:
        key = "purity"
    else:
        key = "particle"
    return {"key": key, "value": val}


def parse_requirement(text):
    t = text.lower()
    quorum = 2 if "dual-sourced" in t else None
    code = re.search(r"requires category c(\d+), grade g(\d+)", t)
    if code:
        return {"key": "c" + code.group(1), "pred": ("label", "g" + code.group(2)), "quorum": quorum}
    m = re.search(r"needs (.+?) of at least ([\d.]+) .* below ([\d.]+)", t)
    if m and m.group(1) in KEY_OF_NAME:
        return {"key": KEY_OF_NAME[m.group(1)], "pred": ("interval", float(m.group(2)), float(m.group(3))),
                "quorum": quorum}
    for g, names in enumerate(MATERIAL_REQ):
        if any(n in t for n in names):
            return {"key": "material", "pred": ("label", "f" + str(g)), "quorum": quorum}
    return None
