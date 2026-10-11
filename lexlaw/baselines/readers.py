"""Slot-by-slot readers (baselines B3, B5 and the hand parser R; method Sections 8 and 12).

A reader answers, for every slot, a mask over its links (or None for "unknown"). It is used
either fixed (`as_fixed`) or as a prior over the unstructured space (`with_prior`, beta = 2).
"""
import json
import math
import os
import re

import numpy as np

from ..envs.text import MATERIAL_REQ, MATERIAL_SUP
from ..model.belief import Belief, blind_space


def tokens(s):
    return set(re.findall(r"[a-z0-9./]+", s.lower()))


def _slots(net):
    for v in sorted(net.slots):
        for j, (links, _) in enumerate(net.slots[v]):
            yield v, j, links


def read_keyword(net, sup, req):
    """B3: links with the maximal Jaccard overlap of lower-cased tokens; all links when it is 0."""
    out = {}
    for v, j, links in _slots(net):
        r = tokens(req[(v, j)])
        sc = np.array([len(r & tokens(sup[int(u)])) / max(len(r | tokens(sup[int(u)])), 1) for u in links])
        out[(v, j)] = sc >= sc.max() - 1e-12 if sc.max() > 0 else np.ones(len(links), bool)
    return out


def read_tfidf(net, sup, req):
    """B3: links whose TF-IDF cosine is at least 0.9 of the slot's maximum; all links when it is 0."""
    docs = list(sup.values()) + list(req.values())
    df = {}
    for d in docs:
        for t in tokens(d):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(len(docs) / c) for t, c in df.items()}

    def vec(s):
        return {t: idf.get(t, 0.0) for t in tokens(s)}

    def cos(a, b):
        num = sum(a[t] * b[t] for t in a if t in b)
        den = math.sqrt(sum(x * x for x in a.values()) * sum(x * x for x in b.values()))
        return num / den if den > 0 else 0.0
    out = {}
    for v, j, links in _slots(net):
        r = vec(req[(v, j)])
        sc = np.array([cos(r, vec(sup[int(u)])) for u in links])
        out[(v, j)] = sc >= 0.9 * sc.max() if sc.max() > 0 else np.ones(len(links), bool)
    return out


def _parse_value(text):
    t = text.lower()

    def num(pat):
        m = re.search(pat, t)
        if m is None:
            raise AttributeError
        return float(m.group(1))
    try:
        if " gpa" in t:
            return num(r"([\d.]+) gpa") * 1000
        if " ksi" in t:
            return num(r"([\d.]+) ksi") / 0.1450377
        if " mpa" in t:
            return num(r"([\d.]+) mpa")
        if "deg f" in t:
            return (num(r"([\d.]+) deg f") - 32) * 5 / 9
        if re.search(r"[\d.]+ k\.", t):
            return num(r"([\d.]+) k\.") - 273.15
        if "deg c" in t:
            return num(r"([\d.]+) deg c")
        if " kv" in t:
            return num(r"([\d.]+) kv") * 1000
        if " mv" in t:
            return num(r"([\d.]+) mv") / 1000
        if re.search(r"[\d.]+ v\.", t):
            return num(r"([\d.]+) v\.")
        if "ppm" in t:
            return 100 - num(r"([\d.]+) ppm") / 1e4
        if "mass fraction" in t:
            return num(r"assay ([\d.]+)") * 100
        if "% pure" in t:
            return num(r"([\d.]+) % pure")
        if " nm" in t:
            return num(r"([\d.]+) nm") / 1000
        if " mm" in t:
            return num(r"([\d.]+) mm") * 1000
        if "micron" in t:
            return num(r"([\d.]+) micron")
    except AttributeError:
        pass
    return None


def read_parser(net, sup, req):
    """R: hand-written parser, slot by slot (reference level for readers; not a method)."""
    fam_sup = {p.lower(): g for g, ps in enumerate(MATERIAL_SUP) for p in ps}
    fam_req = {p.lower(): g for g, ps in enumerate(MATERIAL_REQ) for p in ps}
    out = {}
    for v, j, links in _slots(net):
        r = req[(v, j)].lower()
        m = re.search(r"at least ([\d.]+) .* below ([\d.]+)", r)
        code = re.search(r"category c(\d+), grade g(\d+)", r)
        if code:
            out[(v, j)] = np.array([f"category c{code.group(1)}, grade g{code.group(2)}" in sup[int(u)].lower() for u in links])
        elif m:
            lo, hi = float(m.group(1)), float(m.group(2))
            vals = [_parse_value(sup[int(u)]) for u in links]
            out[(v, j)] = np.array([x is not None and lo <= x < hi for x in vals])
        else:
            g = next((gg for p, gg in fam_req.items() if p in r), None)
            out[(v, j)] = None if g is None else np.array(
                [any(p in sup[int(u)].lower() and gg == g for p, gg in fam_sup.items()) for u in links])
    return out


LLM_PROMPT = """You match purchasing requirements to supplier catalogue entries.
For each item decide which of the offered suppliers satisfy the requirement.
A supplier satisfies a numeric requirement only if its stated value, converted to the requirement's
unit, lies inside the required class (at least the lower bound and below the upper bound).
For material requirements, the catalogue entry must belong to the required material family.
If a requirement says no specification is on file, answer "?" for that item.

{items}

Answer with one JSON object and nothing else, mapping each item number to the list of letters of the
suppliers that satisfy it (possibly empty), or to "?". Example: {{"1": ["a", "c"], "2": [], "3": "?"}}"""


def read_llm(net, sup, req, chat, keys=None, batch=25):
    """B5: a language model answers slot by slot, in batches of 25 (reference read_llm)."""
    keys = keys or [(v, j) for v, j, _ in _slots(net)]
    out = {}
    for b in range(0, len(keys), batch):
        part = keys[b:b + batch]
        lines = []
        for i, (v, j) in enumerate(part, 1):
            links = net.slots[v][j][0]
            lines.append(f"Item {i}. Requirement: {req[(v, j)]}\n" + "\n".join(
                f"   {chr(97 + q)}) {sup[int(u)]}" for q, u in enumerate(links)))
        answer = chat([{"role": "user", "content": LLM_PROMPT.format(items="\n".join(lines))}])
        m = re.search(r"\{.*\}", answer, flags=re.S)
        try:
            parsed = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            parsed = {}
        for i, (v, j) in enumerate(part, 1):
            a = parsed.get(str(i), "?")
            links = net.slots[v][j][0]
            if isinstance(a, list):
                letters = {str(x).strip().lower()[:1] for x in a}
                out[(v, j)] = np.array([chr(97 + q) in letters for q in range(len(links))])
            else:
                out[(v, j)] = None
    return out


def as_fixed(net, answers):
    """Planner input: unknown or empty answers become 'every link is compatible'."""
    return np.concatenate([np.ones(len(links), bool) if (answers.get((v, j)) is None or not answers[(v, j)].any())
                           else answers[(v, j)] for v, j, links in _slots(net)])


def with_prior(net, spaces_placeholder, answers, beta=2.0):
    """Unstructured space with each slot's prior multiplied by exp(-beta * Hamming distance to the answer)."""
    b = Belief(net, spaces_placeholder, use_text=False, escape=False)
    for v in b.cons:
        sl = spaces_placeholder.by_consumer[v]
        c = blind_space([len(net.links_of_slot(s)) for s in sl])
        for j, s in enumerate(sl):
            ans = answers.get((v, j))
            if ans is not None:
                dist = np.array([int((m != ans).sum()) for m in c.masks[j]])
                c.fixed[j] = c.fixed[j] * np.exp(-beta * dist)
        b.sp[v] = c
    return b
