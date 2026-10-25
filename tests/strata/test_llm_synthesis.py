"""Tests of the shared LLM code-loading machinery (no network)."""

import pytest

from strata.llm.synthesis import SynthesisError, extract_code_blocks, load_definitions

ANSWER = """Here is a law:
```python
class Law:
    params = {"g": 2.0}
    def prob(self, f):
        return (f.a >= self.params["g"]).astype(np.float32)
```
and a helper
```python
def helper(x):
    return math.sqrt(x)
```
"""


def test_extract_and_load():
    blocks = extract_code_blocks(ANSWER)
    assert len(blocks) == 2
    ns = load_definitions(blocks[0], ["Law"])
    assert "Law" in ns and ns["Law"].params["g"] == 2.0
    assert load_definitions(blocks[1], ["helper"])["helper"](4.0) == 2.0


def test_failures_are_reported():
    with pytest.raises(SynthesisError):
        load_definitions("def f(:\n  pass", ["f"])
    with pytest.raises(SynthesisError):
        load_definitions("x = 1", ["Law"])
    with pytest.raises(SynthesisError):
        load_definitions("while True:\n    pass", [], timeout=2.0)
