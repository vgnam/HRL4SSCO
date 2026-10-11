# lexlaw

Implementation of the method in [`method.md`](../method.md) (LexLaw: text-grounded symbolic world
models for SSCO with hidden local laws), on the DSR benchmark (dependency-constrained supply-chain
recovery, variants C, Q, A).

Python >= 3.10, numpy (tested with 1.26.4), pytest for the tests. No other dependency for the core.

```
lexlaw/
  envs/dsr.py          Network, make_network, operating (vectorised Op_m), Episode; variants C, Q, A
  envs/text.py         make_texts for levels 1-3; quorum sentence (DSR-Q); level-4 loader
  model/schema.py      program loader (4.1, child-process parsing with a time limit),
                       hypotheses and stated masks (3.2), checks C1-C7 (4.2)
  model/belief.py      prior and channel (3.3), elimination and EM (3.4), quorum, escape (3.5),
                       marginal likelihood L(sigma, eta), key statistics and flags (4.3)
  model/rates.py       rho_hat, phi_hat
  plan/restore.py      restoration-set greedy, with quorum (5.1)
  plan/budget.py       schedule family, aftershock detection, rollout selection (5.2)
  llm/client.py        OpenAI-compatible chat client with an on-disk cache (key from .env, never logged)
  llm/synthesize.py    line sampling, prompts, repair loop (4.1-4.2), revision with the likelihood test (4.3)
  baselines/           B0/B1 (simple.py), B2 and "classes given" (versionspace.py),
                       B3 keyword/TF-IDF, B5 slot-by-slot LLM reader, R hand parser (readers.py)
  programs/            the hand-written reference schema program (labels f0, f1, f2)
  agent.py             one episode of the agent (observe, EM, budget level, restoration greedy)
  procedure.py         Algorithm LexLaw on a chain of episodes (Section 6), with revision
  experiments/         tables.py (Section 9), gates.py (Section 10.4 gates that need no language model)
tests/test_regression.py   Section 10.3
```

WS-option retrained on DSR (baseline B8) lives in [`../DSR`](../DSR), next to the IM and RPP
problems of HRL4SSCO.

## Commands

```bash
py -3.11 -m pytest -q tests/test_regression.py          # Section 10.3, must pass first
py -3.11 -m lexlaw.experiments.tables 9.4               # one table of Section 9 (9.1 9.2 9.4 9.5 9.6 9.7)
py -3.11 -m lexlaw.experiments.gates G5 known15         # gates G3, G4, G5 (C, known15, known30, random)
```

The reference scripts named in the method (`probe_supply*.py`) are not executed or vendored here;
the package reimplements them and is checked against the printed values.

## Status (development networks only; test networks 100-129 not generated)

| stage | gate | result |
|---|---|---|
| 0 | 10.3 regression | **pass**: every value exact (networks, fingerprints, B0, B1, channel + EM + MAP returns and eta, variant P eta) |
| 1 | G1 | **pass**: Table 9.4 and Table 9.5 reproduced to the printed digit, all rows (channel, eta = 0.4 tie, variant P, B2, classes without text) |
| 2 | G2 (language model writes the program) | **not run**: code in `llm/`, needs an API key in `.env` and approval of the calls |
| 3 | G3 (revision flags) | **fails at level 3**: level 2 passes (tensile flagged 12/12, correct program 12/12, bug + escape -2.5% vs B2 -34.1%); level 3: tensile flagged on 6/12 (need >= 10), correct 12/12, bug + escape -13.1% vs B2 -34.1% |
| 4 | G4 (DSR-Q) | **(a) fails**: quorum learner without lines -42.5% -> -14.5% by episode 5 (need within 5 points); **(b) passes**: level-3 lines + quorum sentence improve the first episode by +18.4 points |
| 5 | G5 (rollout budget planner, true model) | **pass**: DSR-A known f=0.15 +13.8% (best static +13.5%), f=0.30 +19.9% (best static +19.8%), DSR-C +2.5%, DSR-A random +12.3% vs asap |
| 6 | baselines, level 4 | B8 (WS-option on DSR) in `../DSR`: trained 20 x 10 episodes on networks 30-99 (the paper's budget), first-episode gap on networks 0-11 x 8: DSR-C -56.4% (B1 -41.6%), DSR-Q -71.5% (B1 -48.6%), DSR-A -62.6% (B1 -39.7%); seed 0, longer pre-training (allowed by G6b) not run. B4, B5b, B6, B7 and level-4 text not done |
| 7 | final runs | not started (gates 2-4 are open) |

Tables 9.1, 9.2, 9.4, 9.5, 9.6 (static rows, run inside G5) and 9.7 are reproduced exactly,
including standard errors and budget used. Table 9.3 needs the cached slot-by-slot answers of the
language model (`supply3_user/`), which are not part of this repository.

Choices not fixed by the method, made here:
- DSR-Q text: the quorum sentence uses its own stream `default_rng(7000 + index)`.
- DSR-A(random): the shock day of an episode is `default_rng(seed + 77).integers(4, 11)`.
- EM sums skip escaped consumers. An escaped consumer whose observation would eliminate every
  link subset ignores that observation (as the reference does).
- G5 is run with the true model and the true failure rate in the simulator (phi known); with the
  estimate phi_hat the day-0 choice uses the prior (1 + 0) / (2 + 0).
