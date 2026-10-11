"""A language model writes the schema program (method Sections 4.1-4.3) [specified, untested].

Call budget per text level: 1 initial call + at most 3 repair calls before any interaction, and at
most 1 revision call per episode. The language model never sets eta, never edits the belief, and
never bypasses the likelihood test.
"""
import math
import re

from ..model.schema import PASS, ProgramError, Schema

INTERFACE = '''A schema program is a Python module with two pure functions. It may import only `re` and `math`;
names containing a double underscore, and open/eval/exec/compile/input/globals/locals/getattr/setattr,
are not allowed. Available builtins: str float int bool len any all sorted set dict list tuple min max
abs round range enumerate zip isinstance and the usual exception classes.

def parse_supplier(text: str) -> dict | None:
    # {"key": str, "value": float | str}
    # key   : identifier of the property the line is about, e.g. "tensile_strength"
    # value : a number in the key's canonical unit, or a label for categorical keys

def parse_requirement(text: str) -> dict | None:
    # {"key": str, "pred": ("interval", lo, hi) | ("label", name), "quorum": 1 | 2 | None}
    # None if the line carries no requirement

The framework decides compatibility with a fixed function:
    satisfies(value, ("interval", lo, hi)) = value is a number and lo <= value < hi
    satisfies(value, ("label", name))      = value == name
so a supplier line and a requirement line about the same property must use the same key, and a
supplier's value must be expressed in the canonical unit that requirement lines use (convert units).
Categorical values must be mapped to the same label on both sides (e.g. a material family).'''

PROMPT = """You write a parser for supplier catalogue lines and purchasing requirement lines.

{interface}

Sample catalogue lines:
{sup}

Sample requirement lines:
{req}

Write the complete program. Answer with one Python code block and nothing else."""

REPAIR = """Your program failed these label-free checks on the development networks:
{report}

Current program:
```python
{source}
```

Fix the program. Answer with one Python code block containing the complete corrected program."""

REVISE = """Interaction data suggests that your program misreads lines about the key {key!r}.
Slots of this key whose stated requirement contradicts what was observed (or whose consumers left
the hypothesis space), with the requirement line, the catalogue lines of the linked suppliers, and
the sets of linked suppliers (by letter) that are still consistent with the observations:
{bad}

Slots of the same key where the stated requirement agrees with the observations:
{good}

Current program:
```python
{source}
```

If a conversion or mapping is wrong, fix it. Answer with one Python code block containing the
complete program."""


def template(line):
    return re.sub(r"[\d.]+", "#", line)


def sample_lines(lines, per_template=3, max_lines=80):
    """Map each line to its template, sort templates, keep the first 3 distinct lines of each."""
    by_t = {}
    for ln in lines:
        lst = by_t.setdefault(template(ln), [])
        if ln not in lst and len(lst) < per_template:
            lst.append(ln)
    out = []
    for t in sorted(by_t):
        out.extend(by_t[t])
    return out[:max_lines]


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return max(m, key=len) if m else text


def _report(chk):
    lines = []
    for c in ("C1", "C2", "C4", "C5", "C7"):
        v = chk[c]
        if v is not None and v < PASS[c]:
            lines.append(f"{c}: rate {v:.3f} below {PASS[c]}; failing examples: {chk['examples'][c][:20]}")
    for k, v in chk["C3"].items():
        if v < PASS["C3"]:
            ex = [e for e in chk["examples"]["C3"]][:20]
            lines.append(f"C3 (key {k}): a parsed supplier value satisfies exactly one requirement class in only "
                         f"{v:.3f} of cases (need {PASS['C3']}); examples (line, number of classes hit): {ex}")
    if not chk["C6"]:
        lines.append(f"C6: exceptions or non-deterministic output: {chk['examples']['C6'][:20]}")
    return "\n".join(lines)


def synthesize(chat, nets_texts, reader_answers=None, max_repairs=3, log=print):
    """Initial call + up to `max_repairs` repairs. Returns (Schema or None, checks, sources tried)."""
    sup_pool = [s for net, sup, req in nets_texts for u, s in sorted(sup.items())]
    req_pool = [r for net, sup, req in nets_texts for k, r in sorted(req.items())]
    prompt = PROMPT.format(interface=INTERFACE, sup="\n".join(sample_lines(sup_pool)), req="\n".join(sample_lines(req_pool)))
    messages = [{"role": "user", "content": prompt}]
    tried, passing = [], []
    for attempt in range(1 + max_repairs):
        source = extract_code(chat(messages))
        try:
            schema = Schema(source)
            chk = schema.checks(nets_texts, reader_answers)
        except (ProgramError, SyntaxError) as e:
            schema, chk = None, None
            report = f"The program could not be loaded: {e}"
        tried.append(source)
        if chk is not None:
            log(f"   attempt {attempt}: C1 {chk['C1']:.3f} C2 {chk['C2']:.3f} C3min {chk['C3min']:.3f} C4 {chk['C4']:.3f} "
                f"C5 {chk['C5']:.3f} C6 {chk['C6']} C7 {chk['C7']} passed {chk['passed']}")
            if chk["passed"]:
                passing.append((schema, chk))
                break
            report = _report(chk)
        messages = messages + [{"role": "assistant", "content": f"```python\n{source}\n```"},
                               {"role": "user", "content": REPAIR.format(report=report, source=source)}]
    if not passing:
        return None, None, tried
    best = max(passing, key=lambda p: (p[1]["C7"] or 0.0, p[1]["C3min"], -len(p[0].source)))
    return best[0], best[1], tried


def revise(chat, schema, belief, key, stats, net, sup_text, req_text, nets_texts, reader_answers=None, log=print):
    """One revision call for a flagged key; accepted iff the new program passes Section 4.2 and
    L(sigma', eta') - L(sigma, eta) >= log 20 on all stored interaction data."""
    def describe(pairs):
        out = []
        for v, j in pairs:
            s = belief.spaces.by_consumer[v][j]
            sl = belief.spaces.slots[s]
            letters = [chr(97 + q) for q in range(len(sl.links))]
            sups = "; ".join(f"{letters[q]}) {sup_text[int(u)]}" for q, u in enumerate(sl.links))
            if belief.sp[v].kind == "schema":
                post = belief.slot_posterior(v, j)
                cons = ["".join(letters[q] for q in range(len(sl.links)) if sl.masks[k][q]) for k, p in post.items() if p > 0]
            else:
                cons = ["(left the program's hypothesis space)"]
            out.append(f"- requirement: {req_text[(v, j)]} | suppliers: {sups} | consistent sets: {cons}")
        return "\n".join(out) or "(none)"
    msg = REVISE.format(key=key, bad=describe(stats[key]["high"]), good=describe(stats[key]["low"]), source=schema.source)
    source = extract_code(chat([{"role": "user", "content": msg}]))
    try:
        new = Schema(source)
        chk = new.checks(nets_texts, reader_answers)
    except (ProgramError, SyntaxError) as e:
        log(f"   revision of key {key!r}: program rejected ({e})")
        return schema, belief, False
    if not chk["passed"]:
        log(f"   revision of key {key!r}: fails Section 4.2")
        return schema, belief, False
    nb = belief.rebuild(new.spaces(net, sup_text, req_text))
    d = nb.log_marginal() - belief.log_marginal()
    ok = d >= math.log(20)
    log(f"   revision of key {key!r}: delta L = {d:.2f} ({'accepted' if ok else 'rejected'})")
    return (new, nb, True) if ok else (schema, belief, False)
