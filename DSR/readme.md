# DSR: supply-chain recovery for WS-option

Dependency-constrained supply-chain recovery from [`method.md`](../method.md) (Section 2), as a third
HRL4SSCO problem next to IM and RPP. The simulator is `lexlaw.envs.dsr` (same networks and random
streams as the LexLaw experiments), so returns are paired with the reference planner B0 and the
heuristic B1.

- 300 nodes in 4 layers (raw suppliers, plants, assemblers, customers); a node operates iff it is up and
  every input slot has a compatible operating supplier. Which links are compatible is hidden.
- Each day (T = 15) the agent repairs a set of down producers, at most `cap` per day and `K` in total;
  a repair succeeds w.p. 0.9, producers fail w.p. 0.01 per day. Reward: weight of served customers.
- Variants: `C` (plain), `Q` (hidden slot quorum, 2.4), `A` (aftershock on day 7 hits 15% of producers, 2.5).

WS-option: higher layer chooses the day's budget, lower layer picks the nodes one at a time; the
first half of the epochs uses the average budget policy (sleep). Training networks are 30-99, test
networks 0-11 x 8 seeds (method.md indexing). Runs on CPU by default (`DSR_DEVICE=cuda` to change).

```bash
python Main.py --variant C        # or Q, A; --epochs 20 --episodes 10 as in the paper
```
