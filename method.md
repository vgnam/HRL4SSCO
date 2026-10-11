LexLaw: Text-Grounded Symbolic World Models for Sequential Stochastic Combinatorial Optimization with Hidden Local Laws
Method and benchmark specification, 2026-10-10. Written for an implementer (Codex) and a critical
reviewer. "LexLaw" is a working name.
Every statement that matters is labelled:
[proved] — a proof is given in this document.
[measured] — a number printed by a script in this folder; the script and its log are named.
[specified, untested] — a design decision that has not been run. Each has a gate in
Section 10.4 or is listed as out of scope in Section 10.6.
[negative result] — something that was tried and did not help. These stay in the paper.
The reference scripts named below are normative: where this text and a script disagree, the
script is right and the text has a bug. Section 12 lists reference behaviours that the text does
not otherwise mention.
Indexing used throughout. Network `i ≥ 0` is generated with seed `100 + i`. Development
networks are `i = 0, …, 29` (most tables use `0, …, 11`; language-model runs use `0, …, 3`).
Training networks for learned baselines are `i = 30, …, 99`. Test networks are `i = 100, …, 129`.
The episode seed for network `i` and run index `s` is `7919·i + 13·s`.
---
0. Summary
Setting. Sequential stochastic combinatorial optimization (SSCO): over `T` days with a total
budget `K`, choose a set of nodes of a graph each day; the graph state then evolves by a stochastic
local law; reward is additive over nodes. HRL4SSCO (WS-option) and LMTA (GoalZero) study two
problems (adaptive influence maximization, stochastic orienteering) whose law is fixed and is
learned implicitly, by a policy or a latent model, from many training episodes.
What earlier work in this project established.
On the published AIM configurations a training-free rule derived from the known cascade law
matches LMTA's reported numbers on three of four configurations and is 6.6% lower on the fourth
(286.64 / 307.80 / 324.16 / 151.07 versus 286.11 / 303.50 / 324.15 / 161.71). Measured in an
earlier session with `exp/stage1_aim.py`; that script is not part of this folder.
When the cascade law is hidden, sparse likelihood selection over a relational expert library
recovers it without a language model, and decisions are insensitive to the remaining model
error [measured], Section 9.8.
A language model asked to write a whole law from raw transition logs succeeded on 1 of 5 laws
[measured], `GRAPHPOE_METHOD.md` Section 9.3.
The question for this document is therefore narrow: in which SSCO problems does a world model
need knowledge that interaction data gives only slowly, and how should a language model supply it?
Answer proposed. Split the hidden part of the world model into
form — how a node's next status is computed from its neighbours;
grounding — which relation instances hold (e.g. which supplier can serve which input slot);
rates — a few scalar probabilities.
Rates are counted. Form is chosen by likelihood from a small hypothesis library. Grounding is
where documents carry information that interaction reveals one trial at a time. The language model
is used in one place: it writes a schema program that parses documents into typed attributes
and predicates (and, in one variant, stated form parameters). The program induces a small
hypothesis space for every hidden relation and a document channel `Pr[text | hidden]`; exact
elimination on interaction data then corrects the text where it is missing or wrong.
New problem family. Dependency-constrained supply-chain recovery (DSR): repair nodes of a
four-layer AND/OR supply network under a budget; which supplier is compatible with which input
slot is hidden and is described by noisy catalogue and requirement text.
Headline measurements [measured] — first episode on a new network, budget 20, return relative
to the same planner given the true relation:
	clean text (level 2)	degraded text (level 3)
no text; hypotheses are subsets of links	−34.1%	−34.1%
keyword / TF-IDF matching, with learning (4 networks)	−37.0% / −36.9%	−36.3% / −36.4%
slot-by-slot reader = LLM, with learning (4 networks)	−1.1%	−21.8%
slot-by-slot reader = hand parser, with learning	0.0%	−19.8%
schema-level model: class hypotheses + document channel + elimination, hand-written reference program	0.0%	−11.0%
Rows without a note are from 12 networks × 8 episodes (Sections 9.3–9.4).
What is not claimed.
That a language model can write the schema program. This is the main untested component
(gate G2). What is measured: (a) a language model reading slot by slot is within 1–3 points of a
hand parser; (b) with a hand-written program, the schema-level model reduces the level-3 gap
from 19.8% to 11.0%.
Any result on text that the authors did not generate (threat T1).
Any comparison with a trained neural world model or with WS-option on DSR (not run).
Novelty. No literature search has been done for "read documents to initialise a world model,
then correct it by interaction" (threat T7).
Optimality. The reference is a greedy planner given the true relation, not the optimal policy;
learners occasionally exceed it.
---
1. Problem class: relational SSCO with a hidden local law (R-SSCO)
1.1 Definition
An instance is `I = (G, T, K, κ, ω, θ*, D)`.
`G = (V, R)`: a finite typed graph; `R` is a set of typed relations (directed edge sets); nodes
and edges may carry attributes.
Horizon `T`, total budget `K`, daily cap `κ`.
State on day `t ∈ {0, …, T−1}`: `s_t = (x_t, B_t, t)`, node statuses `x_t ∈ X^V`, remaining budget
`B_t`, `B_0 = K`.
Action: a set `A_t ⊆ 𝒜(x_t) ⊆ V`, `|A_t| ≤ min(κ, B_t)`; `B_{t+1} = B_t − |A_t|`.
Transition:
```
  primitive step:   y_{t+1}(v) ~ p_{θf, ψ}( · | x_t on N[v],  A_t ∩ N[v],  t ),   independently over v
  closure:          x'_{t+1} = Γ_{θf, θg}( y_{t+1} )                               deterministic
  reward:           r_t = Σ_v ω_v · φ( x_t(v), x'_{t+1}(v), 1[v ∈ A_t] )
  exogenous shock:  x_{t+1} = Ξ_{t+1}( x'_{t+1} )      a random map of known form; the identity unless stated
  ```
`N[v]` is the closed neighbourhood of `v` within a fixed radius. `Γ` computes derived statuses and
is the identity when there are none. The reward of day `t` is paid on the state before the shock.
Return `Σ_{t=0}^{T−1} γ^t r_t`.
Hidden parameter `θ* = (θf*, θg*, ψ*)`: form, grounding, rates.
Documents `D`: text attached to nodes and edges; possibly empty.
The agent knows `G`, `T`, `K`, `κ`, `ω`, the reward function and the hypothesis classes. It reads `D`
before day 0 and observes `x_t` in full every day. It does not know `θ*`.
Assumptions.
A1 (factorisation) primitive statuses are conditionally independent across nodes.
A2 (locality) the law of `v` depends on the state and the action only within a fixed radius.
A3 (observability) `x_t` is observed in full.
A4 (stationarity) `θ*` is constant over all episodes on an instance.
A5 (acyclic closure) `Γ` is the unique solution of local equations on a DAG.
1.2 Protocol and metric
A chain is `E` consecutive episodes on one instance: the initial state is resampled each
episode, `θ*` stays fixed, the agent keeps what it has learned. Cold start is the first episode
of a chain. A method may carry domain-level artefacts (a schema program) from other instances of
the same domain, never instance-level knowledge.
For a policy `π`, the reference `π°` is the same planner given `θ*`. The metric is the paired
relative gap at episode `e`:
```
Gap_e(π) = Σ_{i,c} [ J_{i,c,e}(π) − J_{i,c,e}(π°) ]  /  Σ_{i,c} J_{i,c,e}(π°),
```
over networks `i` and chains `c`, both policies on identical random streams. `Gap` can be positive
because `π°` is not optimal. Standard errors: the tables of Section 9 (development) use the
standard error of the paired differences over (network, seed) pairs, divided by the mean reference
return; final tables (Section 10.4, stage 7) use the standard error over network means.
1.3 Problems in this document
problem	hidden form	hidden grounding	uncertain exogenous part	documents	status
AIM-H (2.6)	activation law	—	—	none	form learned without a language model [measured]
DSR-C (2.1–2.3)	—	required class of every slot	—	levels 1–3	questions Q1 and Q3 answered yes; Q2 yes for the first episodes only [measured]
DSR-Q (2.4)	slot quorum	as DSR-C	—	as DSR-C plus a quorum sentence [specified]	Q1 only [measured]
DSR-A (2.5)	—	as DSR-C	aftershock	as DSR-C	pacing headroom only [measured]
The three probe questions used in Section 9 are: Q1 does planning with a wrong model lose
return; Q2 is a plain learner without text insufficient; Q3 does reading text help. They
are different from the implementation gates G1–G6 of Section 10.4.
---
2. Benchmark problems: exact definitions
2.1 DSR-C: network and hidden relation
Reference generator: `probe_supply.make_instance(numpy.random.default_rng(100 + i))`. The
description is for the reader; reproduce networks by calling the reference (Section 10.3).
Nodes. `V = V_0 ⊔ V_1 ⊔ V_2 ⊔ V_3`, sizes `(90, 90, 80, 40)`: raw suppliers, component plants,
assemblers, customers; indices are consecutive by layer. Producers `P = V_0 ∪ V_1 ∪ V_2`
(`|P| = 260`), customers `C = V_3`.
Producer attributes. Each `u ∈ P` has an output category `a(u) ~ Uniform{0,…,5}` and a class
`g(u) ~ Uniform{0,1,2}`, independent.
Input slots. Each consumer `v ∈ V_i`, `i ∈ {1,2,3}`, has `J_v` slots: `J_v = 1` for customers,
`J_v ~ Uniform{2,3}` otherwise. Slot categories are drawn uniformly without replacement. For slot
`(v, j)` with category `c`:
```
pool   = { u ∈ V_{i−1} : a(u) = c }                 (the slot is dropped if the pool is empty)
N_vj   = uniform random subset of the pool of size min(4, |pool|)       "links", kept in draw order
n_vj   = g(u°),  u° ~ Uniform(N_vj)                                      "required class"
m_vj   ∈ {0,1}^{N_vj},   m_vj(u) = 1[ g(u) = n_vj ]                      "compatibility mask"
```
By construction every slot has at least one compatible link (property F).
Customer weights. `w_c ~ Uniform(1, 3)` for `c ∈ C`; `w_v = 0` for producers.
Hidden. `θg* = (n_vj)`, equivalently the masks `m_vj`. The agent is given the layers, the links
`N_vj` and `w`. It is not given `a`, `g` or `n`.
Network 0: 460 slots, 1840 links, 903 compatible links [measured] (`probe_supply4.py 12 8 0`).
2.2 DSR-C: dynamics, actions, reward
Reference: `probe_supply.run` (true model), `probe_supply4.episode` (learners).
```
T = 15,   ρ = 0.9 (repair success),   φ = 0.01 (daily failure),   down_0 = 0.35,   γ = 1
(K, κ) ∈ { (20, 4), (45, 6) }
```
Status. `up_t ∈ {0,1}^V`; customers are always up. The operating status is derived:
```
op(u) = up(u)                                                         u ∈ V_0
op(v) = up(v) · Π_{j=1..J_v} 1[ Σ_{u ∈ N_vj} m_vj(u) · op(u) ≥ 1 ]     v ∈ V_1 ∪ V_2 ∪ V_3
```
computed layer by layer; write `op = Op_m(up)`. A node operates iff it is up and every slot has at
least one compatible operating supplier.
Initial state. A uniformly random subset of `int(0.35 · 260) = 91` producers is down.
Day `t = 0, …, 14`.
The agent observes `(up_t, op_t = Op_{m*}(up_t), B_t)`.
It chooses `A_t ⊆ { u ∈ P : up_t(u) = 0 }`, `|A_t| ≤ min(κ, B_t)`; `B_{t+1} = B_t − |A_t|`. An
attempted repair is paid for whether or not it succeeds.
Primitive step, independently per node:
```
   v ∈ A_t :        up_{t+1}(v) ~ Bernoulli(ρ)
   v ∈ P \ A_t :    up_{t+1}(v) = up_t(v) · (1 − ζ_v),   ζ_v ~ Bernoulli(φ)
   v ∈ C :          up_{t+1}(v) = 1
   ```
Reward `r_t = Σ_{c ∈ C} w_c · op_{t+1}(c)`, `op_{t+1} = Op_{m*}(up_{t+1})`.
After day 14 the agent sees the final `(up_15, op_15)` (used for learning only). Return
`Σ_{t=0}^{14} r_t`; serving every customer every day would give `15 · Σ_c w_c ≈ 1192`.
Random streams. `rng = default_rng(seed)`; initial down set
`rng.choice(producer indices, 91, replace=False)`; each day `u1 = rng.random(n)` then
`u2 = rng.random(n)` (`n = 300`); repair of `v` succeeds iff `u1[v] < ρ`; `v` fails iff `u2[v] < φ`.
Every policy consumes the stream identically, so comparisons are paired.
R-SSCO reading. `x_t = (up_t, op_t)`; the primitive step is the repair/failure law, of known
form, with rates `ψ = (ρ, φ)` given to the agent; `Γ = Op_m` with hidden grounding `m`.
2.3 DSR-C: documents
Reference: `probe_supply3.make_texts(inst, level, seed = 5000 + i)`. Every producer `u` has a
catalogue line `x_u`; every slot `(v, j)` a requirement line `y_vj`.
Level 1 (codes). `"Category C{a}, grade G{g}."` and `"Requires category C{c}, grade G{n}."`
Level 2 (domain wording). Categories 0–4 are numeric properties with three classes:
cat	property	canonical unit	classes `[lo, hi)`	catalogue wording
0	tensile strength	MPa	250–400, 400–550, 550–700	MPa, GPa, ksi
1	maximum service temperature	deg C	80–150, 150–250, 250–400	deg C, deg F, K
2	rated voltage	V	12–48, 48–240, 240–1000	V, kV, mV
3	purity	%	95–99, 99–99.9, 99.9–99.999	"% pure", "total impurities … ppm", "assay … mass fraction"
4	median particle size	micron	1–10, 10–100, 100–1000	micron, mm, nm
A catalogue value is `lo + (hi − lo) · U(0.08, 0.92)` (purity: `U(0.15, 0.85)`), printed in one of
three wordings chosen uniformly. The requirement line states the class in the canonical unit:
`"Needs {property} of at least {lo} {unit} and below {hi} {unit}; values outside this class are rejected."`
Category 5 is a material with three families; the catalogue names a grade or standard
("AISI 304 sheet", "ASTM A36 plate", "EN AW-6082 bar", …) and the requirement describes the
family functionally ("non-magnetic corrosion-resistant stainless steel", …). Tables:
`probe_supply3.BANDS`, `MATERIAL_SUP`, `MATERIAL_REQ`. Order of random draws: as in the reference.
Level 3 (degraded). Level 2, except that each requirement line independently is replaced by
`"No specification on file."` with probability 0.25, and otherwise states the wrong class
`(n + U{1,2}) mod 3` with probability 0.15 (0.20 of the lines that state a class). Catalogue
lines are unchanged. Measured on the 12 development networks: 24.9% of lines missing; 19.65% of
stated lines wrong (`logs/supply4_part0.log`).
Level 4 (third-party text) [specified, untested; required for the paper]. Lines written by a
party that has not seen the readers (a different model, or a person given only property, value
and class), with free paraphrase, distractor attributes and inconsistent units; or real datasheet
lines. Until level 4 exists, no claim about "real text" is allowed.
2.4 DSR-Q: hidden quorum (a hidden form parameter)
Reference: `probe_supply5.py`. Same network as DSR-C. Each slot has a quorum `q_vj ∈ {1, 2}`:
```
for consumers in index order, slots in order:
    eligible = v ∈ V_1 ∪ V_2 and |m_vj| ≥ 2
    q_vj = 2 if eligible and rng_q.random() < p_dual else 1        (the draw is made only when eligible)
rng_q = default_rng(900 + i),   p_dual = 0.3

op(v) = up(v) · Π_j 1[ Σ_{u ∈ N_vj} m_vj(u) · op(u) ≥ q_vj ].
```
Hidden: `q` and `n`. The agent knows that customers have `q = 1`, and knows `p_dual`.
Documents [specified, untested]. DSR-C text; additionally a requirement line of a slot with
`q = 2` ends with `" Dual-sourced: two qualified suppliers must be running."` At level 3 the
sentence is dropped with probability `δ_drop = 0.15` when `q = 2` and added with probability
`δ_add = 0.05` when `q = 1`; a line replaced by "No specification on file." carries no sentence.
2.5 DSR-A: aftershock (a problem where pacing the budget matters)
Reference: `probe_supply4.AFTERSHOCK = (D, f)`. At the start of day `D` — after the reward of day
`D − 1` has been paid and before the observation of day `D` — every producer independently goes
down with probability `f` (stream `default_rng(seed + 424243)`, one `random(260)` draw, identical
for all policies). Documents as DSR-C.
DSR-A(known): `D = 7`, `f ∈ {0.15, 0.30}`, both given to the agent. Measured (Section 9.6).
DSR-A(random) [specified, untested]: `D ~ Uniform{4, …, 10}` unknown, `f = 0.2` given.
A second axis is the daily failure rate `φ ∈ {0.01, 0.03, 0.05}`.
2.6 AIM-H: adaptive influence maximisation with a hidden activation law
Defined in `GRAPHPOE_METHOD.md`, Sections 1 and 7; summarised here. Directed graph; node status in
{inactive, active, removed}; seeding set `A_t` of inactive nodes, `Σ_t |A_t| ≤ K`; each inactive
unseeded node activates independently with probability `p*(v; G, s_t ⊕ A_t)`; active nodes become
removed; `r_t = |A_t| + #newly activated`; `γ = 0.99`. `p*` is hidden and drawn from ten law families.
No documents. In R-SSCO terms: hidden form, known grounding.
---
3. The world model
`M = (σ, F, b, ψ)`: a schema program `σ`, a form hypothesis `F`, a belief `b` over the grounding, and
rates `ψ`. Sections 3.1–3.5 instantiate it for DSR; Section 3.6 covers AIM-H.
3.1 Schema program
`σ` is a Python module with two pure functions; the framework supplies `satisfies`.
```python
def parse_supplier(text: str) -> dict | None:
    # {"key": str, "value": float | str}
    # key   : identifier of the property the line is about, e.g. "tensile_strength"
    # value : a number in the key's canonical unit, or a label for categorical keys
def parse_requirement(text: str) -> dict | None:
    # {"key": str, "pred": ("interval", lo, hi) | ("label", name), "quorum": 1 | 2 | None}
    # None if the line carries no requirement

def satisfies(value, pred) -> bool:          # fixed, not written by the language model
    # ("interval", lo, hi): isinstance(value, (int, float)) and not isinstance(value, bool) and lo <= value < hi
    # ("label", name):      value == name
```
The program holds the domain knowledge: unit conversions, and which catalogue names belong to
which family. It never sees the hidden relation.
3.2 From program to hypotheses
Given `σ`, the documents and the links, with no interaction data:
```
(k_u, z_u)        = parse_supplier(x_u), or ⊥
R_vj              = parse_requirement(y_vj) = (k_vj, p_vj, q̂_vj), or ⊥
key(v, j)         = the most common k_u over u ∈ N_vj        (ties: the key of the earliest such link)
if R_vj ≠ ⊥ and k_vj ≠ key(v, j):   treat the line as missing (R_vj ← ⊥) and count a C2 failure (4.2)

𝒫_k               = the distinct predicates p_vj over slots with R_vj ≠ ⊥ and key(v, j) = k      (pooling)
                    in canonical order: intervals sorted by (lo, hi), then labels sorted as strings
K_vj              = |𝒫_key(v,j)|
μ_p(u)            = 1[ k_u = key(v, j) and satisfies(z_u, p) ]            for p ∈ 𝒫_key(v,j),  u ∈ N_vj
unresolved_vj(u)  ⇔ μ_p(u) = 0 for every p ∈ 𝒫_key(v,j)

ℳ_vj  (ordered)   = the non-zero masks μ_p for p in canonical order, duplicates removed keeping the first,
                    followed by the singletons e_u for unresolved u in link order
μ̂_vj              = μ_{p_vj}  if R_vj ≠ ⊥,  else ⊥                         "stated mask" (may be the zero mask)
```
`e_u` is the indicator of the single link `u`. `ℳ_vj` is never empty.
Why pool. A slot whose line is missing or wrong still has at most `K_vj` candidate masks,
because the classes of its key were read from other slots. On the development networks the mean
number of joint hypotheses per consumer is 8.57 (max 27), against 1473.8 (max 3375) when a
hypothesis is an arbitrary non-empty subset of the links [measured] (`logs/supply4_part0.log`).
Reference note. `probe_supply4.read_schema` and `Belief` use class indices instead of masks and
drop unresolved suppliers. The two coincide when the predicates of a key partition the values and
every supplier is resolved, which holds for DSR with the reference program. Class indices follow
the canonical order for intervals and the family index 0, 1, 2 for materials.
3.3 Prior and document channel
```
base prior:        π⁰_vj(μ) = |μ| / Σ_{μ' ∈ ℳ_vj} |μ'|

document channel:  λ_vj(μ ; η) = 1                       if R_vj = ⊥ or K_vj < 2
                               = 1 − η                    if μ = μ̂_vj
                               = η / (K_vj − 1)           otherwise
```
Generative reading: a stated predicate is the true one with probability `1 − η` and otherwise one
of the other `K_vj − 1` predicates of the key, uniformly. A stated predicate whose mask is zero
cannot be true (property F); every candidate then gets the same factor and the slot falls back
to the base prior. A consumer hypothesis is `h = (μ_1, …, μ_{J_v}) ∈ ℋ_v = ×_j ℳ_vj`, enumerated in
the order of `itertools.product` over the ordered `ℳ_vj`, with unnormalised weight
```
W_v(h ; η) = Π_j π⁰_vj(h_j) · λ_vj(h_j ; η).
```
For DSR level 3 this model is the generator's with `η = 0.2`: the base prior is the generator's
(a slot requires the class of a uniformly chosen link) and the channel is the generator's noise.
That exactness is a property of the benchmark, not of the method (threat T10).
Variant P (the form used in the first runs of Sections 9.4–9.5). A stated class with no linked
supplier is ignored, and the slot prior is
`(1 − η)·1[μ = μ̂] + η·π⁰(μ)/(1 − π⁰(μ̂))·1[μ ≠ μ̂]`. Both forms are implemented in
`probe_supply4.Belief` (`form="prior"` / `form="channel"`); their returns do not differ
(Section 9.4). The channel form is the specified one.
3.4 Likelihood, posterior, and the estimate of η
For a consumer `v` that is up on day `s`, with observed supplier statuses `o^s = op_s` on `N(v)`:
```
f_v(h ; o)  = Π_j 1[ ⟨h_j, o on N_vj⟩ ≥ 1 ]                         predicted operating status
b_v^t(h)    ∝ W_v(h ; η) · Π_{s ≤ t : up_s(v) = 1} 1[ f_v(h ; o^s) = op_s(v) ].
```
Proposition 1 [proved] (exact, factorised posterior). Consider DSR-C or DSR-Q under A3. Let
the prior over joint hypotheses `h = (h_v)_v ∈ ×_v ℋ_v` be a product over consumers that gives
positive mass to the true relation, and let the policy choose `A_s` as a function of the history
`(up_{0:s}, op_{0:s}, A_{0:s−1})`, the documents and independent randomness only. Then for fixed `η`
the posterior given the history up to day `t` is `Π_v b_v^t(h_v)`.
Proof. Given `h`, the probability of the history is a product of: (i) the initial-state
probability, free of `h`; (ii) policy factors, free of `h` by assumption; (iii) primitive
transitions `Pr[up_{s+1} | up_s, A_s]`, which depend on `ρ, φ` (and the shock law) only; (iv)
`Pr[op_s | up_s, h] = 1[op_s = Op_h(up_s)]`. The network is a layered DAG, so `op = Op_h(up)` holds
iff the local equation `op(v) = up(v) · f_v(h_v; op on N(v))` holds at every node; hence
`1[op_s = Op_h(up_s)] = Π_v 1[op_s(v) = up_s(v) · f_v(h_v; op_s on N(v))]`. The factor of `v` involves
`h_v` only and equals 1 for every `h_v` when `up_s(v) = 0`. Multiplying by a product prior and
normalising — possible because the truth has positive mass and is consistent with every
observation — gives a product posterior with factors `b_v^t`. ∎
Scope. (a) The reference policy `π°` uses the true relation and is excluded; it is not a learner.
(b) With the schema hypothesis space the product prior is the generator's (requirements are drawn
independently per slot; level-3 noise is independent per slot). In the unstructured space of
Section 3.5 the product prior is an approximation: masks of different slots that share suppliers
are coupled through the suppliers' classes. (c) `η` is estimated, so the method uses a plug-in
posterior; in the exact Bayesian treatment slots are coupled through `η`.
Consequences: inference is exact given the model, costs `O(Σ_v |ℋ_v| · J_v)` per day, and needs no
sampling and no reasoning about paths.
Estimate of η (MAP-EM). With `S` the set of slots with `R_vj ≠ ⊥` and `K_vj ≥ 2`:
```
E-step:   r_vj = b_v( h_j ≠ μ̂_vj )         (r_vj = 1 when μ̂_vj is the zero mask)
M-step:   η ← ( a + Σ_{(v,j) ∈ S} r_vj ) / ( a + b + |S| ),          a = 1,  b = 4
```
Proposition 2 [proved]. For a fixed data set, the iteration is EM for the posterior mode of
`η` under a `Beta(a+1, b+1)` prior, with an exact E-step; the log posterior of `η` does not decrease.
Proof. Let `z_vj = 1[the stated predicate of (v, j) is wrong]`. Given `η`, the complete-data log
posterior is `(a + Σ z) log η + (b + Σ (1 − z)) log(1 − η) − Σ z log(K_vj − 1) + const`, because the
observation likelihood is 0/1 and free of `η`. Its expectation under the current posterior
replaces `z_vj` by `r_vj`, exact by Proposition 1; the maximiser is the M-step. Monotonicity is the
standard EM property. ∎
Defaults: `η = 0.2` initially; three iterations after each day's observation (none after the final
observation of an episode); `η` is kept across the episodes of a chain. The data grow daily, so
monotonicity applies within a day's three iterations, not across days.
3.5 Extensions of the belief
Quorum (DSR-Q) [specified, untested]. A slot hypothesis is `(μ, q)`, `q ∈ {1, 2}`. Prediction:
`f_v(h; o) = Π_j 1[⟨μ_j, o⟩ ≥ q_j]`. Weight of a slot hypothesis:
```
π⁰_vj(μ) · λ_vj(μ ; η) · Pr[q | μ] · λ^q_vj(q)

Pr[q = 2 | μ] = p_dual · 1[ |μ| ≥ 2 ] · 1[ v ∉ C ],          Pr[q = 1 | μ] = 1 − Pr[q = 2 | μ]
λ^q_vj(q)     = 1                                              if R_vj = ⊥
              = 1 − δ_drop if q = 2 else δ_add                 if the program returned quorum = 2
              = δ_drop     if q = 2 else 1 − δ_add             if it returned quorum = 1 or None
```
Defaults `δ_drop = 0.15`, `δ_add = 0.05`. Updates, over the slots with `R_vj ≠ ⊥`, with `s_vj = 1` when
the program returned quorum = 2 and posterior probabilities `b_v(q_j = ·)`:
```
δ_drop ← ( 1.5 + Σ (1 − s_vj) · b_v(q_j = 2) ) / ( 10 + Σ b_v(q_j = 2) )
δ_add  ← ( 0.5 + Σ s_vj · b_v(q_j = 1) )       / ( 10 + Σ b_v(q_j = 1) )
```
(pseudo-counts chosen so that the values with no data are 0.15 and 0.05). Proposition 1 covers the
quorum case: its proof uses only that `f_v` is local.
Rates. `ρ̂ = (1 + successes) / (2 + attempts)`,
`φ̂ = (1 + failures) / (2 + producer-days at risk)`. A producer-day `(u, t)` is at risk when `u` is up
on day `t`, is not repaired on day `t`, and day `t + 1` is not a day on which an aftershock occurs
(known `D`) or is detected (Section 5.2). Used only by the budget planner's simulator.
Escape to the unstructured space [specified, untested]. If an observation eliminates every
hypothesis of `ℋ_v`, the program is wrong about `v` (the truth is outside `ℋ_v`). Then `v` switches
permanently to the space of all non-empty link subsets per slot with the independent prior
`Pr[link compatible] = 1/3` (`probe_supply2.VersionSpace(kind="blind")`), replays its stored
observations, and is recorded as an escaped consumer for Section 4.3.
Marginal likelihood of a program. By the factorisation in Proposition 1,
```
L(σ, η) := log Pr[ op_{0:t} | up_{0:t}, σ, η ]
         = Σ_v log [ (1 − ζ) · m_v^schema + ζ · m_v^blind ],        ζ = 10⁻³,
m_v^schema = Σ_{h ∈ ℋ_v^σ} W̄_v(h ; η) · 1[h consistent with v's observations]       (W̄ = W normalised over ℋ_v)
m_v^blind  = the same sum over the unstructured space with its prior.
```
`L` is the score used to compare programs (Section 4.3) [specified, untested].
3.6 The form part, and AIM-H
In DSR-C the form (AND over slots, OR over suppliers) is given. In DSR-Q one form parameter per
slot is hidden and sits inside the belief. In AIM-H the form is an unknown activation law, learned
by the competing-risk product of experts of `GRAPHPOE_METHOD.md`:
```
p_w(v) = 1 − exp( −( w₀ + Σ_k w_k H_k(v) ) ),    w ≥ 0,
```
with `H_k` from a typed relational grammar, forward selection with convex refits and a BIC stop.
AIM-H has no grounding to read and no schema program. The two learners are not combined in any
problem of this document; a problem that hides an open-ended form and a grounding is future work.
---
4. Writing the schema program with a language model [specified, untested]
4.1 Inputs, loader, call budget
Input to the model. The interface of Section 3.1 and a sample of lines; nothing about the
hidden relation. Sampling: pool the catalogue lines of development networks 0–2, map each line to
its template (every maximal run of digits and decimal points replaced by `#`), sort the
templates, take the first 3 distinct lines of each, at most 80 lines; the same for requirement
lines.
Loader. A new loader is needed (the one in `synth_law.py` rejects `import`, `try` and `lambda`
and has no `re`). Rules: parse the source with `ast`; the only import statements allowed are
`import re` and `import math`, and they are removed from the tree before execution, with the `re`
and `math` modules placed in the namespace instead (the restricted builtins have no `__import__`);
reject names containing `__`, and the names `open`, `eval`, `exec`, `compile`, `input`, `globals`,
`locals`, `getattr`, `setattr`; execute with builtins restricted to
`str, float, int, bool, len, any, all, sorted, set, dict, list, tuple, min, max, abs, round, range, enumerate, zip, isinstance, ValueError, TypeError, AttributeError, IndexError, KeyError, Exception`;
require the two entry points. Run parsing in a child process with a wall-clock limit of 30 s per
1000 lines (this works on Windows; `signal.SIGALRM` does not). This guards against accidents; it is
not a security sandbox.
Call budget. One program per text level, written once from the sample above: 1 initial call +
at most 3 repair calls before any interaction, + at most 1 revision call per episode. For
comparison the slot-by-slot reader costs about 19 calls and 68k input / 25k output tokens per
network [measured] (glm-5.3, low reasoning effort, Section 9.3).
4.2 Label-free checks (before any interaction)
Evaluated on development networks 0–2. Each check returns a rate and up to 20 failing examples,
which are sent back in a repair call.
check	statement	pass level
C1 coverage	catalogue lines parsed to a key and a value	≥ 0.98
C2 key agreement	all linked suppliers of a slot share one key, equal to the requirement's key when present	≥ 0.98 of slots
C3 partition	a parsed supplier value satisfies exactly one predicate of `𝒫` for its key (zero or several both fail); computed per key	≥ 0.95 for every key
C4 feasibility	a stated predicate has a non-zero mask (property F)	≥ 0.85 of stated slots (reference program: 100% at level 2, 94.5% at level 3)
C5 discrimination	the stated mask is a non-zero proper subset of the links	≥ 0.50 of stated slots
C6 purity	same output on repeated calls; no exception; within the time limit	all
C7 agreement with a slot-level reader	on 100 random slots with a requirement, the stated mask equals the answer of the slot-by-slot language-model reader (Section 9.3)	≥ 0.90 at every level
C2 uses the graph: links of a slot share a category, so a parser that gives them different keys is
wrong, with no label needed. C3 is computed per key because one wording is about a third of a
key's suppliers but only about 1/18 of all suppliers: an omitted conversion that throws one wording
outside every class fails its key, while a global rate would not see it. A conversion error that
lands values inside another class (an omitted deg F conversion, for instance) is not caught by
C3; it is left to C7 and to Section 4.3. C5 rejects the degenerate program that maps everything to
one key and one always-true predicate (which would pass C1–C4). C3 and C5 rely on classes being
disjoint and discriminating, a property of DSR; in another domain they are reported, not enforced.
C7 costs four calls. When a language model is available it is a gate like the others ("passes
Section 4.2" means C1–C7); when none is available it is skipped. Selection among programs that
pass: the highest C7; ties by the highest minimum of C3 over keys, then the shortest source.
4.3 Revision from interaction
After each episode, with `η̂` the current estimate and `r_vj` as in Section 3.4, for every key `k`:
```
S_k    = { (v, j) ∈ S : key(v, j) = k,  v not escaped }
c_k    = Σ_{(v,j) ∈ S_k} r_vj
S_{−k} = { (v, j) ∈ S : key(v, j) ≠ k,  v not escaped }
η_{−k} = ( a + Σ_{(v,j) ∈ S_{−k}} r_vj ) / ( a + b + |S_{−k}| )            (same r; EM is not re-run)
e_k    = number of escaped consumers that have a slot of key k
N_k    = number of consumers that have a slot of key k
```
Flag key `k` when either
(contradictions) `|S_k| ≥ 20`, `c_k ≥ 5` and `c_k / |S_k| > η_{−k} + 3 · sqrt( η_{−k}(1 − η_{−k}) / |S_k| )`; or
(escapes) `e_k ≥ 2` and `e_k / N_k` is the largest among keys.
Random text errors are spread over keys at rate `η`; a parsing error concentrates in one key. For a
flagged key, call the model with the current program, up to 20 slots of that key with `r_vj > 0.9`
or belonging to escaped consumers (their lines and the masks still consistent), and 20 slots of
the key with `r_vj < 0.1`. Accept the returned program `σ'` iff it passes Section 4.2 and
```
L(σ', η̂') − L(σ, η̂) ≥ log 20       on all interaction data stored so far,
```
where `η̂'` is the EM estimate after rebuilding the belief under `σ'` and replaying the stored
observations. The language model never sets `η`, never edits the belief, and never bypasses the
likelihood test.
4.4 Fallback
If no program passes Section 4.2 after the repair calls: use the slot-by-slot reader extended to
return which linked suppliers are interchangeable (baseline B5b), and build `ℳ_vj` from its answers
without pooling. If that is not available: the slot-by-slot reader with the unstructured space
(`probe_supply3.with_prior`, `β = 2`), the configuration that loses 21.8% at level 3.
---
5. Planning
Two levels, as in HRL4SSCO and LMTA: how much to spend today, and on which set.
5.1 Low level: restoration-set greedy
Reference: `probe_supply.plan_model`; with quorum, `probe_supply5.plan_q`. Input: `up`, a point model
`m̂` (and `q̂`), and `room`, the number of repairs allowed today.
```
up' ← up;  A ← ∅
repeat while room > 0:
    op^m ← Op_m̂(up')
    for each customer c in index order with op^m(c) = 0:   R(c) ← Restore(c)
    c* ← the first maximiser of  w_c / |R(c)|  over { c : R(c) ≠ ⊥, 0 < |R(c)| ≤ room }
    if none: stop
    A ← A ∪ R(c*);  up'(u) ← 1 for u ∈ R(c*);  room ← room − |R(c*)|
return A

Restore(v):                                                    memoised within one pass of the loop
    if op^m(v) = 1: return ∅
    S ← {v} if up'(v) = 0 else ∅
    for each slot j of v:
        cand ← [ Restore(u) for u ∈ N_vj in link order with m̂_vj(u) = 1 and Restore(u) ≠ ⊥ ]
        if |cand| < q̂_vj: return ⊥                              (q̂_vj = 1 in DSR-C)
        S ← S ∪ (the q̂_vj smallest sets of cand; earlier link first on ties)
    return S
```
Per-slot choices are made independently and then united ("tree approximation"); the cost is the
size of the union. Planned repairs are assumed to succeed. The planner is a heuristic with no
guarantee. It is the same for every model, so differences between rows of Section 9 are due to the
model alone. (The reference function also takes the true served vector; it does not affect the
result.)
Model given to the planner. For every consumer, the first maximiser of `b_v` in the enumeration
order of Section 3.3 (MAP). The agent replans every day after updating the belief.
[negative result] Planning with the posterior instead of its mode (`probe_supply4.py 12 8 1`,
variant P): one posterior sample per consumer per day (Thompson sampling) loses 7.8% at level 2
and 16.1% at level 3, against 0.0% and 11.2% for MAP; choosing and ranking restoration sets by
marginal compatibility (`plan_scored`) loses 3.0% and 13.7%. Neither is used.
5.2 High level: budget schedule chosen by rollouts of the model
Schedule family. `asap`, and `(r, mode)` for reserve `r ∈ {⌊0.15K⌋, ⌊0.3K⌋, ⌊0.5K⌋, ⌊0.75K⌋}`
(3, 6, 10, 15 for `K = 20`) and `mode ∈ {hold, uniform}`:
```
asap:                                  b_t = min(κ, B_t)
(r, mode), not released and B_t > r:   b_t = min(κ, B_t − r)
(r, uniform), otherwise:               b_t = min(κ, B_t, ⌈B_t / (T − t)⌉)
(r, hold), not released, B_t ≤ r:      b_t = 0
(r, hold), released:                   b_t = min(κ, B_t)

released on day t ⇔ an aftershock has occurred by day t, or t ≥ T − 3
   DSR-A(known):   occurred ⇔ t ≥ D
   DSR-A(random):  detected on the first day t with
                   #{ u ∈ P : up_{t−1}(u) = 1, up_t(u) = 0 } > φ̂·N + 4·sqrt(φ̂·N) + 3,   N = #{ u ∈ P : up_{t−1}(u) = 1 }
   DSR-C:          never
```
`(r, hold)` is the "reserve r" schedule of Section 9.6. `(⌊0.5K⌋, uniform)` is its
"half-then-uniform" schedule except that it does not overshoot the reserve on the switching day.
Algorithm (model-predictive choice within the family) [specified, untested]. On day 0, every
third day, and on the day an aftershock occurs or is detected:
```
for each schedule θ in the family:
    V(θ) ← mean over n_sim = 8 simulated continuations from the current state, in the model,
            following θ and the low-level planner with m̂; common random numbers across θ
θ* ← argmax V(θ)       ties: asap first, then smaller r, then hold
```
The simulator is the environment of Section 2 with `m̂` in place of `m*` (certainty equivalence),
rates `(ρ, φ̂)` and the aftershock law of the variant: none (DSR-C); day `D` with fraction `f` if it
has not occurred (DSR-A known); `D` uniform over the days of `{4, …, 10}` after today, if none has
been detected (DSR-A random). Measured headroom of this level, with the true model and static
schedules (Section 9.6): at most +1.6% in DSR-C, +5.7% at `φ = 0.03`, +13.5% and +19.8% in DSR-A. In
DSR-C the high level may be fixed to `asap`.
---
6. Full procedure
```
Algorithm LexLaw (one network, one chain of E episodes)
 0. σ ← the domain's program (reuse if it exists; else Sections 4.1–4.2; else the fallback of 4.4)
 1. Build ℳ_vj, μ̂_vj, π⁰, K_vj (3.2–3.3);  η ← 0.2;  b_v ← normalised W_v
 2. for episode e = 1..E:
      for day t = 0..T−1:
         observe (up_t, op_t);  store and apply the observation for every up consumer (3.4, escape 3.5)
         three EM steps for η
         on day 0, every third day, or when an aftershock occurs or is detected: choose the schedule θ* (5.2)
         b_t ← schedule;  m̂ ← MAP;  A_t ← restoration-set greedy(up_t, m̂, b_t) (5.1);  act
         update ρ̂, φ̂
      observe the final state; store and apply it (no EM step)
      revise σ if a key is flagged and the new program passes the likelihood test (4.3);
      if σ changed: rebuild the hypotheses and replay all stored observations
```
Stored per consumer: every pair `(op on N(v), op(v))` observed while `v` was up.
---
7. Relation to prior work
source	what is kept	what changes
HRL4SSCO / WS-option	two decision levels: budget, then set	both levels are computed from an explicit model instead of trained options
LMTA / GoalZero	plan with a world model; a separate budget decision	the model is a program plus an exact posterior, not a latent network; it is usable in the first episode
WorldCoder	a language model writes code; code is revised from counterexamples; code that explains more data replaces code that explains less	the code is a parser of documents, not the transition function; acceptance is an exact marginal likelihood
PoE-World	a model assembled from many small programs with fitted weights	used for the form part only (GraphPoE, AIM-H)
TheoryCoder / bilevel planning	an abstract level plans, a low level executes	the abstract level is the budget schedule
Weaknesses of the two SSCO methods that the design targets: they cannot use documents; they need
many episodes under one law before acting well; the learned law cannot be inspected. Whether these
weaknesses cost return on DSR has not been measured for those methods (baselines B7, B8).
---
8. Baselines (all required for the paper)
id	baseline	status
B0	reference planner given the true model	measured
B1	law-free heuristic: repair the down nodes with the largest downstream customer weight (`plan_reach`)	measured: −36% to −46%
B2	no text, unstructured version space, prior 1/3 per link (`learn-blind`)	measured; the prior parameter was not tuned (T13)
B3	keyword overlap; TF-IDF cosine (`read_keyword`, `read_tfidf`)	measured
B4	sentence-embedding similarity with a threshold fitted on 50 labelled slots	to do
B5	slot-by-slot language-model reader, fixed and as a prior	measured (levels 2 and 3, 4 networks)
B5b	slot-by-slot reader that also returns which linked suppliers are interchangeable; hypotheses of 3.2 without pooling	to do
B6	a language model writes the whole transition function from logs, revised on counterexamples (WorldCoder-style, no documents)	to do
B7	graph neural network world model trained on interaction data + the same planner (stand-in for LMTA, whose code is not public)	to do
B8	WS-option (public code) retrained on DSR	to do
R	hand-written parser, slot by slot and as a schema program (reference level for readers; not a method)	measured
B4 decides whether a language model is needed at level 4. B5b separates pooling from the rest of
the schema-level model. B7 and B8 decide the sample-efficiency claim.
---
9. Measurements so far
Development networks only (0–29 in 9.1–9.2, 0–3 in 9.3, 0–11 elsewhere); budget `(20, 4)` unless stated. "±" is the standard error of the paired
differences over (network, seed) pairs as a percentage of the mean reference return; seeds within a
network are not independent (threat T4). Each run in 9.3, 9.4, 9.6 and 9.7 starts from a fresh
belief.
9.1 Q1 — does a wrong model cost return? [measured] `probe_supply.py 30 3` and `probe_supply.py 30 3 20 4`; logs `supply1_K45.log`, `supply1_K20.log`
model given to the planner	K = 45	K = 20
true relation (return)	1094.3	871.8
true relation with 20% of the entries flipped	−30.4%	−30.4%
ignores classes (any link is believed compatible)	−71.9%	−64.8%
law-free heuristic B1	−38.3%	−42.8%
The "ignores classes" rows are inflated at both budgets: that planner believes most customers are
already served and spends 2.6 budget units.
9.2 Q2 — is a plain learner without text enough? [measured] `probe_supply2.py 30 5`; log `supply2.log`
One chain of five episodes per network, K = 20:
learner	ep. 1	ep. 2	ep. 3	ep. 4	ep. 5
supplier classes given, required class hidden	−20.1%	−10.8%	−9.3%	−3.5%	−1.0%
nothing given; hypotheses are link subsets (B2)	−32.8%	−24.8%	−21.5%	−11.3%	−9.8%
fixed model with 20% of entries flipped	−31.4%	−25.6%	−34.0%	−29.9%	−26.2%
heuristic B1	−42.9%	−45.8%	−39.9%	−44.0%	−39.1%
At K = 45: −9.1% → −0.1% with classes given, −24.1% → −3.1% for B2. A learner that is given
supplier classes closes the gap in four to five episodes; the learner with no information is still
at −9.8% after five. The cost of not reading is concentrated in the first episodes.
9.3 Q3 — does reading help, and can a language model read? [measured] `probe_supply3.py`, 4 networks × 8 first episodes, reference return 849.4; log `supply3_replay_llm.log`
Language model: glm-5.3, low reasoning effort; 154 calls, 540,665 input and 199,321 output tokens
for both levels (from the run on the author's machine); the cached answers are in `supply3_user/`
and are replayed without any call by `replay3.py`.
reader	L2: links right	L2: fixed	L2: + learning	L3: links right	L3: fixed	L3: + learning
keyword	50.7%	−61.8%	−37.0%	50.3%	−63.5%	−36.3%
TF-IDF	51.1%	−61.4%	−36.9%	50.7%	−64.4%	−36.4%
hand parser	100%	0.0%	0.0%	85.1%	−27.9%	−20.6%

language model, slot by slot	99.1%	−2.7% ±0.7	−1.1% ±0.5	84.3%	−29.2% ±2.1	−21.8% ±2.0
no text (B2)			−36.0% ±1.9			−36.0% ±1.9
"Fixed" plans with the reader's answer and never revises it (an unknown or empty answer is replaced
by "every link is compatible"); "+ learning" uses the answer as a prior over link subsets
(`with_prior`, `β = 2`). "Links right" is computed over the slots the reader answered; at level 3
the parser and the language model decline about 25% of slots and the keyword readers none, so that
column is not comparable across those rows. Language-model accuracy by category at level 2:
tensile 99.5%, temperature 99.3%, voltage 99.3%, purity 97.3%, particle size 100%, material 99.1%.
At level 3 it answers "?" on 468 of the 478 missing lines, and on the 1378 slots that both it and
the parser answer, the two answers are identical in 96.7% (96.2% of all 1385 slots that carry
text). The loss at level 3 comes from the text, not from the reading.
9.4 The schema-level model [measured] `probe_supply4.py 12 8 1` and `… 5`; logs `supply4_part1.log`, `supply4_part5.log`; reference return 849.0
The schema program is the hand-written reference (`read_schema`): it gives a class to 100% of
suppliers and partitions the links of 100% of slots as the truth.
configuration (MAP plan unless stated)	level 2	level 3
no text, link subsets (B2)	−34.1% ±1.2	−34.1% ±1.2
slot-by-slot answers of the hand parser as a prior over link subsets	0.0%	−19.8% ±1.2
class hypotheses, requirement lines ignored	−21.8% ±1.1	−21.8% ±1.1
class hypotheses + document channel, η by EM	0.0%	−11.0% ±0.9
channel, η fixed at 0.05 / 0.2 / 0.4 (0.4 is on a tie, see below)	0.0% / 0.0% / −3.5% ±0.7	−11.0% / −11.0% / −12.5% ±1.0
variant P, η = 0.2	0.0%	−11.2% ±0.9
variant P, η fixed at 0.05 / 0.4; η by EM	0.0% / 0.0%; 0.0%	−11.2% / −11.7%; −11.2%
variant P, η = 0.2, Thompson sampling [negative]	−7.8% ±0.9	−16.1% ±1.0
variant P, η = 0.2, marginal-scored plan [negative]	−3.0% ±0.9	−13.7% ±1.0
The schema-level model moves level 3 from −19.8% to −11.0%. The two rows differ in three things
at once — the hypothesis space (at most 3 classes instead of 15 link subsets per slot), the text
model (channel instead of `exp(−2 · Hamming)`), and what a slot with a missing line knows — and
they are not separated here. Baseline B5b separates them.
Channel form minus variant P, paired: +0.20 ± 0.30 points at level 3, 0.00 at level 2. No
difference in return; the channel form is specified because it is a generative model of the text.
The estimate of `η` after one episode is 0.192 at level 3 (19.65% of stated lines are wrong) and
0.004 at level 2 (none are).
Sensitivity to a fixed `η` is a threshold, not a slope. Before any observation, a stated class
held by one of four links beats an unstated class held by the other three iff
`(1/4)(1 − η) > (3/4)(η/2)`, i.e. `η < 0.4`. Below 0.4 the return does not depend on `η`; at
`η = 0.4` the two weights are equal in exact arithmetic and the reference's choice is decided by
floating-point rounding, which is what the −3.5% and −12.5% entries show. Estimating `η` avoids
having to place it on the right side of the threshold.
The target set before this probe was "below −10% at level 3"; −11.0% misses it.
9.5 Over episodes, level 3 [measured] `probe_supply4.py 12 8 2` and `… 5`; logs `supply4_part2.log`, `supply4_part5.log`; one chain per network
learner	ep. 1	ep. 2	ep. 3	ep. 4	ep. 5
no text, link subsets (B2)	−38.4% ±3.3	−25.9% ±3.1	−23.8% ±4.2	−13.4% ±2.7	−11.5% ±3.3
class hypotheses, requirement lines ignored	−21.3% ±2.6	−11.4% ±3.8	−9.7% ±2.7	−3.7% ±2.0	−2.0% ±2.2
class hypotheses + channel, EM	−13.2% ±2.4	−8.8% ±3.7	−2.5% ±2.0	−2.9% ±2.0	−1.6% ±1.2
variant P, EM	−14.3% ±2.9	−9.0% ±3.8	−1.5% ±2.8	−2.4% ±2.0	−1.6% ±1.2
With 12 chains per row and no paired test between learner rows, differences after episode 1 are
not established; the largest is at episode 3 (−9.7 ± 2.7 against −2.5 ± 2.0). Any claim about
reading must be stated as a cold-start claim until more chains are run. The estimate of `η` rises
from 0.185 to 0.195 over the five episodes (channel form).
9.6 Does pacing the budget matter? [measured] `probe_supply4.py 12 8 3` and `… 4`; logs `supply4_part3.log`, `supply4_part4.log`; true model
Gap relative to "spend as soon as possible":
environment	uniform	half-then-uniform	reserve 6 until the shock	reserve 10 until the shock
DSR-C, K = 20	−7.0% ±0.8	+1.6% ±0.7		
DSR-C, K = 45	−8.2% ±0.3	−0.8% ±0.2		
φ = 0.03	−1.5% ±1.4	+5.7% ±1.2		
φ = 0.05	−5.3% ±1.7	+3.7% ±1.4		
DSR-A, D = 7, f = 0.15	+0.9% ±1.2	+9.3% ±0.9	+13.5% ±1.1	+11.7% ±1.2
DSR-A, D = 7, f = 0.30	+2.8% ±1.1	+11.9% ±0.9	+18.1% ±1.2	+19.8% ±1.3
In DSR-C the budget level has almost nothing to decide; it matters in DSR-A. These are static
schedules that know the shock day; the rollout planner of Section 5.2 has not been run. A fourth
static schedule (hold 4 units until day 10, then uniform) never beat the best alternative in its
row, "as soon as possible" included: −1.4%, 0.0%, −2.7%, −5.2%, +2.8%, +5.5% in the six rows.
9.7 Hidden quorum, Q1 only [measured] `probe_supply5.py 12 8 0.3`; log `supply5.log`; compatibility known
21% of plant and assembler slots are dual-sourced.
model given to the planner	K = 20 (reference 613.1)	K = 45 (reference 923.9)
believes every slot needs one supplier	−34.4% ±1.8	−38.5% ±1.5
believes every slot that can be dual-sourced is	−31.7% ±1.7	−24.5% ±0.9
A wrong quorum costs about as much as a wrong relation. At K = 45 the first row is partly inflated
in the same way as in 9.1 (36.5 of 45 budget units spent, against 44.6). Q2 and Q3 have not been
run.
9.8 Form learning on AIM-H [measured] `proto_model.py`, `proto_e2e.py`, `ext_test.py --selftest`, `proto_e2e_x.py`; logs `proto_model.log`, `proto_e2e_a.log`, `proto_e2e_b.log`, `ext_selftest.log`, `proto_e2e_x_a.log`, `proto_e2e_x_b.log`
Forward selection over 310 relational experts recovers 8 of 8 hidden activation laws with 1–2
experts and ≥ 99.8% of the likelihood gap; acting with the learned law is within noise of acting
with the true law on 6 of 6 laws tested. On six laws written outside the grammar the base grammar
closes 54–91% of the likelihood gap, and the return lost against the true law is at most
11.8 ± 4.4 points (law X5; the other five are within ±4). No language model is involved.
---
10. Implementation plan
10.1 Package layout
```
lexlaw/
  envs/dsr.py            Network, make_network (wraps the reference), Op, Episode; variants C, Q, A
  envs/text.py           make_texts for levels 1–3 (wraps the reference); quorum sentence; level-4 loader
  model/schema.py        program loader (4.1); hypotheses and stated masks (3.2); checks C1–C7 (4.2)
  model/belief.py        prior and channel (3.3), elimination and EM (3.4), quorum and escape (3.5), L(σ, η)
  model/rates.py         ρ̂, φ̂
  plan/restore.py        restoration-set greedy, with quorum (5.1)
  plan/budget.py         schedule family, aftershock detection, rollout selection (5.2)
  llm/client.py          chat client with an on-disk cache (from synth_law.Chat)
  llm/synthesize.py      line sampling, prompts, repair and revision loops (4.1–4.3)
  baselines/             B1–B8 and R
  experiments/           one script per table of Section 9, plus the new ones
  tests/
reference/               the probe_*.py scripts and logs/, unchanged
```
Python ≥ 3.10, numpy, scipy; nothing else for the core. All file writes use `encoding="utf-8"`.
The API key is read from `.env` and is never printed, logged, cached or committed.
10.2 Interfaces
```python
# envs/dsr.py
@dataclass
class Network: index; n; layer; idx; slots; w; grade; cat; quorum      # slots[v] = [(links, mask), ...]
def make_network(index: int, variant="C", p_dual=0.3) -> Network        # generator seed = 100 + index
def operating(net, up, compat, quorum=None) -> np.ndarray
class Episode:                                                          # reference random streams
    def __init__(self, net, seed, K, cap, T=15, rho=0.9, phi=0.01, aftershock=None)
    def observe(self) -> tuple[np.ndarray, np.ndarray, int, int]        # up, op, budget, day
    def step(self, repairs: list[int]) -> float                         # reward of the day

# model/schema.py
class Schema:
    def __init__(self, source: str)
    def spaces(self, net, sup_text, req_text) -> SlotSpaces             # ordered ℳ_vj, μ̂_vj, π⁰, K_vj, q̂_vj, keys
    def checks(self, nets, texts) -> dict                               # C1–C6: rate and failing examples

# model/belief.py
class Belief:
    def __init__(self, net, spaces: SlotSpaces, eta=0.2, quorum=False)
    def observe(self, up, op) -> None                                   # stores, eliminates, escapes
    def em(self, a=1.0, b=4.0, iters=3) -> float
    def map_model(self) -> dict                                         # consumer -> masks (and quorums)
    def log_marginal(self) -> float                                     # L(σ, η)
    def key_statistics(self) -> dict                                    # per key: |S_k|, c_k, η_{−k}, e_k, N_k, examples
    def rebuild(self, spaces) -> "Belief"                               # replay stored observations

# plan
def restore_greedy(net, up, model, room) -> list[int]
def choose_schedule(simulate, state, model, family, n_sim=8, seed=0) -> tuple
```
10.3 Regression tests (must pass before any new experiment)
Everything is deterministic given the seeds. The values below were produced with Python 3.13 and
numpy 2.5.3; pin the numpy version, and if another version changes the random streams, regenerate
the values with the reference scripts and say so. Budget `(20, 4)`; each row starts from a fresh
belief with `η = 0.2`; "η" is the value after the EM step of day 14.
quantity	value
network 0: slots / links / compatible links / Σw	460 / 1840 / 903 / 80.470782
network 1: slots / links / compatible links / Σw	451 / 1804 / 922 / 74.807786
network 0, seed 0: reference planner / heuristic B1	986.408087 / 492.684857
network 0, seed 13: reference planner / heuristic B1	832.535903 / 516.516856
network 1, seed 7919: reference planner / heuristic B1	875.787169 / 445.115720
network 1, seed 7932: reference planner / heuristic B1	1031.194982 / 457.222355
network 0, level 3, channel + EM + MAP, seed 0: return, η (variant P: η)	840.565890, 0.187624 (0.167993)
network 0, level 3, channel + EM + MAP, seed 13	853.676581, 0.177944 (0.150039)
network 1, level 3, channel + EM + MAP, seed 7919	826.589313, 0.144067 (0.138196)
network 1, level 3, channel + EM + MAP, seed 7932	1017.535203, 0.126149 (0.113176)
Variant P gives the same four returns. Fingerprint of a network: md5 over, for consumers in
increasing index and slots in order, the links as int64 bytes then the mask as uint8 bytes,
followed by `numpy.round(w, 6)` (the full length-300 weight vector) as float64 bytes: network 0 →
`ac10e68a189eec46a7bd083bdfc70766`, network 1 → `02e3baa659df6f861594bc0ad96dd738`.
Then reproduce every table of Sections 9.1, 9.2 and 9.4–9.7 to the printed digit by running the
reference scripts, and 9.3 from the cached answers with `replay3.py supply3_user --table`.
10.4 Stages and gates
Test networks 100–129 must not be generated or run before stage 7. A "point" is one percentage
point of the mean reference return.
stage	work	gate (stop and report if it fails)
0	package the reference; regression tests	10.3 passes exactly
1	mask-form hypotheses, channel, belief, EM, marginal likelihood, escape	G1: with the reference parser rewritten against the 3.1 interface (family labels `"f0"`, `"f1"`, `"f2"`, so the canonical order equals the reference's), the channel rows of Table 9.4 for η by EM, η = 0.05 and η = 0.2, and the channel row of 9.5, are reproduced to the printed digit. Compute weights as plain float64 products in slot order. The η = 0.4 row (a tie) and the variant-P rows (reference only) are not part of the gate
2	language model writes the schema program; checks; repair loop	G2: one program per level, at most 4 calls each, no labels. On networks 0–11: links partitioned as the truth in ≥ 95% of slots at level 2, and first-episode gap within 2 points of the reference program at levels 2 and 3 (0.0% and −11.0%)
3	revision from interaction	G3: at levels 2 and 3, with the ksi conversion removed from the reference program, the tensile key is flagged after one episode on ≥ 10 of 12 networks; with the correct program no key is flagged on ≥ 11 of 12; with the bug and the escape rule, the first-episode gap is better than B2's on the same networks
4	DSR-Q: quorum in the belief; quorum sentence in the text	G4: (a) class hypotheses × quorum with requirement lines ignored reaches within 5 points of the reference by episode 5; (b) adding level-3 requirement lines improves that learner's first-episode gap by ≥ 5 points
5	DSR-A: rates, simulator, schedule choice	G5: with the true model, the rollout planner is within 2 points of the best static schedule of 9.6 in both DSR-A(known) settings and ≥ 8 points above `asap` at f = 0.15; in DSR-C it is at most 1 point below `asap`; in DSR-A(random) it is ≥ 5 points above `asap`
6	baselines B4, B5b, B6, B7, B8; level-4 text	G6a: at level 4, LexLaw's first-episode gap is better than B4's by ≥ 5 points, else the claim that a language model is needed is dropped. G6b: B7 and B8 may be pre-trained on training networks 30–99 for any number of episodes (reported); on an evaluated network they see only the chain's own episodes. At every episode e ≤ 5 of a chain, LexLaw's gap is better than theirs by ≥ 5 points, else the sample-efficiency claim is dropped; run B7 and B8 for 50 episodes to locate the crossover
7	final runs on networks 100–129: 4 chains of 5 episodes per network (chain c, episode e: run index `s = 5c + e`), all problems and levels	none; report everything, including failures
If G2 fails, the method falls back to B5b, and the paper's claim changes from "the language model
writes a reusable program" to "the language model reads slot by slot, at a cost linear in the
number of slots".
10.5 Reporting rules
Every table states networks, seeds, and the script and log that produced it.
Paired differences against the reference planner on identical streams; standard errors over
network means in the final tables.
Negative results stay in the paper: Thompson sampling, marginal-scored planning, pacing in DSR-C,
the −10% target missed at level 3, the language model failing to write laws from raw logs.
No tuning on test networks. Any constant changed after stage 7 starts is reported.
10.6 Out of scope for the first implementation (no gate)
Noisy observation of the operating status; hidden links; an open-ended hidden form combined with a
hidden grounding; a text model in which the catalogue lines are also wrong; transfer of a program
to a second domain.
---
11. Limitations and threats to validity
T1 Self-generated text. Levels 1–3 and the hand parser were written by the same person; the
parser's 100% is by construction. Level 4 is required before any claim about text.
T2 One language model, low reasoning effort, 4 networks for every language-model number.
T3 The reference planner is a greedy heuristic. Gaps are relative to it, not to the optimum.
T4 Dependence. Standard errors in Section 9 treat episodes of one network as independent.
T5 Deterministic closure. Exact elimination relies on the operating status being a
deterministic function of the neighbours' statuses. Under noisy observation the 0/1 likelihood
becomes a soft one; untested.
T6 Given structure. Links and slots are known; only compatibility (and quorum) is hidden. A
reviewer can call DSR-C "entity matching plus a planner". DSR-Q and DSR-A are the answer offered;
neither is fully tested.
T7 Novelty unchecked. To be searched: language-conditioned and manual-reading reinforcement
learning; language-model entity and schema matching; WorldCoder, PoE-World, OneLife, TheoryCoder,
AgentOWL; network restoration and supply-chain recovery in operations research; Bayesian
experimental design on bills of materials.
T8 "Class hypotheses, requirement lines ignored" is not text-free. It uses catalogue text and
the pooled requirement predicates to form classes. It isolates the value of per-slot statements.
T9 Pacing headroom was measured with the true model, static schedules and a known shock day.
T10 The model matches the generator. The base prior and the channel are exactly the level-3
generator's. Ablate with a uniform base prior and with text noise that is not uniform over the
other classes.
T11 The first-episode gap at level 3 has no known lower bound. −11.0% may be close to what
any method can do with 25% of lines missing and 15% wrong, or far from it.
T12 The main component is untested. Nothing in Section 9 uses a program written by a
language model.
T13 Baseline B2's prior (1/3 per link, against a true share of 49% compatible links) was not
tuned; its most probable hypothesis before any observation is "only the first link". Ablate over
{1/3, 1/2, 2/3}. The same space is LexLaw's escape and fallback.
T14 The comparison −19.8% → −11.0% confounds three changes (Section 9.4).
---
12. Reference behaviours not stated above
An implementer needs these to reproduce the numbers, or must change them deliberately.
`Belief.observe` and `VersionSpace.observe` skip a consumer that has a single remaining
hypothesis, and ignore an observation that would eliminate every hypothesis. The package must
store every observation and apply the escape rule instead; with the reference program the two
behaviours give the same returns because the truth is always in the space.
The reference drops unresolved suppliers from every hypothesis instead of adding singleton
masks; with the reference program no supplier is unresolved.
`probe_supply4.episode` runs the EM steps after each day's observation and not after the final
one; `η` and the belief persist across the episodes of a chain when the same object is reused.
Thompson sampling uses its own stream `default_rng(seed + 99991)`, one draw per consumer per
day, in consumer order.
`plan_scored` chooses, within a slot, the believed-compatible link `u` minimising
`|R(u)| − ℓ(u)`, where `ℓ(u)` is the log-probability accumulated along `R(u)` plus the log of the
link's marginal compatibility (clipped below at 10⁻⁶), and ranks customers by
`w_c · exp(ℓ) / |R(c)|`.
`plan_reach` (B1) propagates customer weight upstream, splitting a consumer's weight equally
among the links of each slot, and repairs the down nodes with the largest total.
`as_fixed` (the "fixed" columns of 9.3) replaces an unknown or all-zero answer by all ones.
`with_prior` multiplies the weight of a link-subset hypothesis by `exp(−2 · Hamming distance)`
to the reader's answer for each answered slot.
`read_keyword` marks the links with the maximal Jaccard overlap of lower-cased tokens
`[a-z0-9./]+`; `read_tfidf` marks the links whose TF-IDF cosine is at least 0.9 of the slot's
maximum; both mark every link when the maximum is 0.
In `make_texts`, catalogue lines are generated for producers in index order, then requirement
lines for consumers in index order and slots in order, from one generator; within
`supplier_text` a uniform is drawn, then the wording index, and for purity a second uniform
(materials: one index only).
`add_quorum` consumes a random number only for eligible slots (Section 2.4).
In the channel form the reference includes every slot with a stated class in the EM sums and
uses `max(K_vj, 2)`; in DSR every key has three classes, so this equals the definition of `S`.