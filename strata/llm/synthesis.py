"""Shared machinery for LLM-written code in STRATA (L1 laws, L2 operators).

The LLM answers with Python code blocks that follow a template. This module
extracts the blocks, checks them in a child process with a time limit (so a
syntax error, an infinite loop or a crash at import time cannot take the agent
down), and only then executes them in-process in a namespace that provides
``np`` and ``math``. The returned objects are then validated numerically by
the caller (shapes, ranges, likelihood, non-dominance).
"""

import os
import re
import subprocess
import sys
import textwrap

_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_CHECK = textwrap.dedent(
    """
    import math, sys
    import numpy as np
    src = sys.stdin.read()
    ns = {"np": np, "math": math, "__name__": "strata_llm"}
    exec(compile(src, "<llm>", "exec"), ns)
    missing = [n for n in sys.argv[1:] if n not in ns]
    if missing:
        print("missing: " + ", ".join(missing))
        sys.exit(3)
    print("ok")
    """
)


class SynthesisError(RuntimeError):
    """LLM code that cannot be used (syntax, import, timeout, missing names)."""


def extract_code_blocks(text):
    """Python code blocks of an LLM answer, in order (the whole text if none)."""
    blocks = [b.strip() for b in _BLOCK.findall(text or "")]
    blocks = [b for b in blocks if b]
    if blocks:
        return blocks
    stripped = (text or "").strip()
    return [stripped] if stripped else []


def check_in_subprocess(code, required_names=(), timeout=10.0, python=None):
    """Execute ``code`` in a child process; raise SynthesisError on failure."""
    cmd = [python or sys.executable, "-I", "-c", _CHECK] + list(required_names)
    try:
        proc = subprocess.run(cmd, input=code, capture_output=True, text=True,
                              timeout=timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        raise SynthesisError("timeout after %.0f s while loading the code" % timeout) from None
    if proc.returncode != 0:
        err = (proc.stdout + proc.stderr).strip().splitlines()
        raise SynthesisError("code failed to load: " + (err[-1] if err else "exit %d" % proc.returncode))


def load_definitions(code, required_names=(), timeout=10.0):
    """Check ``code`` in a child process, then exec it here; return its namespace."""
    check_in_subprocess(code, required_names, timeout=timeout)
    import math

    import numpy as np

    ns = {"np": np, "math": math, "__name__": "strata_llm"}
    exec(compile(code, "<llm>", "exec"), ns)
    return {k: v for k, v in ns.items() if not k.startswith("__") and k not in ("np", "math")}
