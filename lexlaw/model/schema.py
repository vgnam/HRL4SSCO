"""Schema programs: loader (method 4.1), hypotheses and stated masks (3.2), label-free checks (4.2)."""
import ast
import hashlib
import math
import multiprocessing as mp
import os
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

ALLOWED_BUILTINS = ("str", "float", "int", "bool", "len", "any", "all", "sorted", "set", "dict", "list", "tuple",
                    "min", "max", "abs", "round", "range", "enumerate", "zip", "isinstance", "ValueError",
                    "TypeError", "AttributeError", "IndexError", "KeyError", "Exception")
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "input", "globals", "locals", "getattr", "setattr"}
REFERENCE_PROGRAM = os.path.join(os.path.dirname(os.path.dirname(__file__)), "programs", "dsr_reference.py")


class ProgramError(Exception):
    pass


def check_source(source):
    """Static rules of the loader; returns the tree with the allowed imports removed."""
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ("re", "math") or (a.asname not in (None, a.name)):
                    raise ProgramError(f"import of {a.name} is not allowed")
            continue
        body.append(node)
    tree.body = body
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ProgramError("imports are only allowed at module level: import re, import math")
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        for nm in names:
            if "__" in nm:
                raise ProgramError(f"name {nm!r} is not allowed")
            if nm in FORBIDDEN_NAMES:
                raise ProgramError(f"name {nm!r} is not allowed")
    return tree


def load_program(source):
    """Executes a schema program under restricted builtins; returns (parse_supplier, parse_requirement)."""
    tree = check_source(source)
    import builtins
    env = {"__builtins__": {k: getattr(builtins, k) for k in ALLOWED_BUILTINS}, "re": re, "math": math}
    exec(compile(tree, "<schema program>", "exec"), env)
    for name in ("parse_supplier", "parse_requirement"):
        if not callable(env.get(name)):
            raise ProgramError(f"missing entry point {name}")
    return env["parse_supplier"], env["parse_requirement"]


def satisfies(value, pred) -> bool:
    """Fixed predicate semantics (Section 3.1); not written by the language model."""
    if pred[0] == "interval":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and pred[1] <= value < pred[2]
    if pred[0] == "label":
        return value == pred[1]
    return False


def _norm_sup(out):
    if not isinstance(out, dict) or not isinstance(out.get("key"), str):
        return None
    v = out.get("value")
    if isinstance(v, bool) or not isinstance(v, (int, float, str, type(None))):
        v = None
    if isinstance(v, (int, float)) and not math.isfinite(float(v)):
        v = None
    return (out["key"], float(v) if isinstance(v, (int, float)) else v)


def _norm_req(out):
    if not isinstance(out, dict) or not isinstance(out.get("key"), str):
        return None
    p = out.get("pred")
    if not isinstance(p, (tuple, list)) or not p:
        return None
    if p[0] == "interval" and len(p) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in p[1:]):
        pred = ("interval", float(p[1]), float(p[2]))
    elif p[0] == "label" and len(p) == 2 and isinstance(p[1], str):
        pred = ("label", p[1])
    else:
        return None
    q = out.get("quorum")
    return (out["key"], pred, q if q in (1, 2) else None)


def _parse_worker(source, sup_lines, req_lines):
    ps, pr = load_program(source)
    res_s, res_r, errors = [], [], []
    for kind, fn, lines, out, norm in (("sup", ps, sup_lines, res_s, _norm_sup), ("req", pr, req_lines, res_r, _norm_req)):
        for line in lines:
            try:
                out.append(norm(fn(line)))
            except Exception as e:                       # noqa: BLE001 - reported by check C6
                out.append(None)
                errors.append((kind, line, f"{type(e).__name__}: {e}"))
    return res_s, res_r, errors


_PARSE_CACHE = {}


def parse_lines(source, sup_lines, req_lines, sandbox=True, seconds_per_1000=30.0):
    """Parse lines with a program in a child process with a wall-clock limit (Section 4.1)."""
    key = hashlib.sha1(("\0".join([source] + list(sup_lines) + ["\1"] + list(req_lines))).encode("utf-8")).hexdigest()
    if key in _PARSE_CACHE:
        return _PARSE_CACHE[key]
    if not sandbox:
        res = _parse_worker(source, list(sup_lines), list(req_lines))
    else:
        limit = max(5.0, seconds_per_1000 * (len(sup_lines) + len(req_lines)) / 1000)
        ctx = mp.get_context("spawn")
        with ctx.Pool(1) as pool:
            job = pool.apply_async(_parse_worker, (source, list(sup_lines), list(req_lines)))
            try:
                res = job.get(timeout=limit)
            except mp.TimeoutError:
                pool.terminate()
                raise ProgramError(f"parsing exceeded the time limit of {limit:.0f} s")
    _PARSE_CACHE[key] = res
    return res


def _pred_order(p):
    return (0, p[1], p[2], "") if p[0] == "interval" else (1, 0.0, 0.0, p[1])


@dataclass
class SlotSpace:
    v: int
    j: int
    s: int                          # flat slot id
    links: np.ndarray
    key: object                     # key(v, j) or None
    masks: list                     # ordered M_vj (bool arrays over the links)
    prior: np.ndarray               # base prior pi0 over masks
    stated: object                  # None (no usable line) or index into masks, -1 for the zero mask
    K: int                          # number of pooled predicates of the key
    quorum_stated: object           # 2, 1 or None as returned by the program; None also when R is missing
    text: str = ""
    pred: object = None


@dataclass
class SlotSpaces:
    slots: list                     # SlotSpace per flat slot id
    by_consumer: dict               # v -> list of flat slot ids
    pools: dict                     # key -> ordered predicate list
    sup: dict                       # u -> (key, value) or None
    c2_failures: int = 0
    unresolved: int = 0
    stats: dict = field(default_factory=dict)


class Schema:
    def __init__(self, source: str, sandbox=True):
        self.source = source
        self.sandbox = sandbox
        if sandbox:                                       # static rules and entry points; never executed here
            tree = check_source(source)
            defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            missing = {"parse_supplier", "parse_requirement"} - defs
            if missing:
                raise ProgramError(f"missing entry point(s) {sorted(missing)}")
        else:
            load_program(source)

    @classmethod
    def reference(cls, sandbox=False, transform=None):
        with open(REFERENCE_PROGRAM, encoding="utf-8") as f:
            src = f.read()
        return cls(transform(src) if transform else src, sandbox=sandbox)

    def parse(self, net, sup_text, req_text):
        us = sorted(sup_text)
        ks = [(v, j) for v in sorted(net.slots) for j in range(len(net.slots[v]))]
        res_s, res_r, errors = parse_lines(self.source, [sup_text[u] for u in us], [req_text[k] for k in ks], self.sandbox)
        return dict(zip(us, res_s)), dict(zip(ks, res_r)), errors

    def spaces(self, net, sup_text, req_text) -> SlotSpaces:
        sup, req, _ = self.parse(net, sup_text, req_text)
        return build_spaces(net, sup, req, req_text)

    def checks(self, nets_texts, reader_answers=None) -> dict:
        return run_checks(self, nets_texts, reader_answers)


def _slot_key(links, sup):
    keys = [sup[int(u)][0] if sup.get(int(u)) is not None else None for u in links]
    cnt = Counter(k for k in keys if k is not None)
    if not cnt:
        return None
    top = max(cnt.values())
    return next(k for k in keys if k is not None and cnt[k] == top)


def build_spaces(net, sup, req, req_text=None) -> SlotSpaces:
    """Hypothesis spaces per slot from parsed lines (Section 3.2)."""
    pre, c2 = [], 0
    for v in sorted(net.slots):
        for j, s in enumerate(net.slots_of[v]):
            links = net.links_of_slot(s)
            key = _slot_key(links, sup)
            R = req.get((v, j))
            if R is not None and R[0] != key:
                R, c2 = None, c2 + 1
            pre.append((v, j, s, links, key, R))
    pools = {}
    for v, j, s, links, key, R in pre:
        if R is not None:
            pools.setdefault(key, set()).add(R[1])
    pools = {k: sorted(ps, key=_pred_order) for k, ps in pools.items()}
    slots, by_consumer, unresolved = [], {}, 0
    for v, j, s, links, key, R in pre:
        P = pools.get(key, [])
        ku = [sup.get(int(u)) for u in links]
        mu = [np.array([x is not None and x[0] == key and satisfies(x[1], p) for x in ku], bool) for p in P]
        masks = []
        for m in mu:
            if m.any() and not any(np.array_equal(m, x) for x in masks):
                masks.append(m)
        unres = [q for q in range(len(links)) if not any(m[q] for m in mu)]
        unresolved += len(unres)
        for q in unres:
            e = np.zeros(len(links), bool)
            e[q] = True
            masks.append(e)
        sizes = np.array([m.sum() for m in masks], float)
        prior = sizes / sizes.sum()
        stated, qs, pred = None, None, None
        if R is not None:
            pred = R[1]
            hat = mu[P.index(R[1])]
            stated = next((i for i, m in enumerate(masks) if np.array_equal(m, hat)), -1) if hat.any() else -1
            qs = R[2]
        sp = SlotSpace(v, j, s, links, key, masks, prior, stated, len(P), qs,
                       req_text.get((v, j), "") if req_text else "", pred)
        slots.append(sp)
        by_consumer.setdefault(v, []).append(s)
    return SlotSpaces(slots, by_consumer, pools, sup, c2, unresolved)


# ----------------------------------------------------------------------------- checks (Section 4.2)
PASS = {"C1": 0.98, "C2": 0.98, "C3": 0.95, "C4": 0.85, "C5": 0.50, "C7": 0.90}


def run_checks(schema, nets_texts, reader_answers=None, n_examples=20):
    """nets_texts: list of (net, sup_text, req_text). reader_answers: optional list (one per network)
    of {(v, j): bool mask or None} from a slot-by-slot reader, for C7."""
    c1 = [0, 0]; c2 = [0, 0]; c4 = [0, 0]; c5 = [0, 0]; c7 = [0, 0]
    c3 = {}
    ex = {k: [] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")}
    c6_ok = True
    for i, (net, sup_text, req_text) in enumerate(nets_texts):
        sup, req, errors = schema.parse(net, sup_text, req_text)
        _PARSE_CACHE.clear()                              # C6 purity: parse once more, uncached
        sup_b, req_b, _ = schema.parse(net, sup_text, req_text)
        if sup_b != sup or req_b != req or errors:
            c6_ok = False
            ex["C6"].extend(errors[:n_examples])
        for u, x in sup.items():
            c1[1] += 1
            ok = x is not None and x[1] is not None
            c1[0] += ok
            if not ok and len(ex["C1"]) < n_examples:
                ex["C1"].append(sup_text[u])
        sp = build_spaces(net, sup, req, req_text)
        for slot in sp.slots:
            keys = {sup[int(u)][0] if sup.get(int(u)) is not None else None for u in slot.links}
            R = req.get((slot.v, slot.j))
            agree = len(keys) == 1 and None not in keys and (R is None or R[0] in keys)
            c2[1] += 1
            c2[0] += agree
            if not agree and len(ex["C2"]) < n_examples:
                ex["C2"].append((req_text[(slot.v, slot.j)], [sup_text[int(u)] for u in slot.links]))
            if slot.stated is not None:
                c4[1] += 1
                c4[0] += slot.stated >= 0
                if slot.stated < 0 and len(ex["C4"]) < n_examples:
                    ex["C4"].append((slot.text, [sup_text[int(u)] for u in slot.links]))
                hat = slot.masks[slot.stated] if slot.stated >= 0 else np.zeros(len(slot.links), bool)
                proper = hat.any() and not hat.all()
                c5[1] += 1
                c5[0] += proper
                if not proper and len(ex["C5"]) < n_examples:
                    ex["C5"].append((slot.text, [sup_text[int(u)] for u in slot.links]))
            if reader_answers is not None and i < len(reader_answers):
                ans = reader_answers[i].get((slot.v, slot.j))
                if ans is not None and slot.stated is not None:
                    hat = slot.masks[slot.stated] if slot.stated >= 0 else np.zeros(len(slot.links), bool)
                    c7[1] += 1
                    same = np.array_equal(hat, np.asarray(ans, bool))
                    c7[0] += same
                    if not same and len(ex["C7"]) < n_examples:
                        ex["C7"].append((slot.text, [sup_text[int(u)] for u in slot.links], list(map(int, ans))))
        for u, x in sup.items():
            if x is None or x[1] is None or x[0] not in sp.pools:
                continue
            hits = sum(satisfies(x[1], p) for p in sp.pools[x[0]])
            c3.setdefault(x[0], [0, 0])
            c3[x[0]][1] += 1
            c3[x[0]][0] += hits == 1
            if hits != 1 and len(ex["C3"]) < n_examples:
                ex["C3"].append((sup_text[u], hits))
    rate = lambda a: a[0] / a[1] if a[1] else 1.0  # noqa: E731
    out = {"C1": rate(c1), "C2": rate(c2), "C3": {k: rate(v) for k, v in c3.items()}, "C4": rate(c4),
           "C5": rate(c5), "C6": c6_ok, "C7": rate(c7) if c7[1] else None, "examples": ex}
    out["C3min"] = min(out["C3"].values()) if out["C3"] else 0.0
    out["passed"] = (out["C1"] >= PASS["C1"] and out["C2"] >= PASS["C2"] and out["C3min"] >= PASS["C3"]
                     and out["C4"] >= PASS["C4"] and out["C5"] >= PASS["C5"] and out["C6"]
                     and (out["C7"] is None or out["C7"] >= PASS["C7"]))
    return out
