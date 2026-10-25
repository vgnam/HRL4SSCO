"""Central configuration for every STRATA layer.

The plan requires one source of truth for hyper-parameters.  Each milestone
adds fields here instead of sprinkling constants through the code.  All
dataclasses are plain (mutable) containers so a CLI can override individual
fields, and ``as_dict`` makes them easy to serialise next to experimental
outputs.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


@dataclass
class IMConfig:
    """Influence-maximisation environment defaults (IM/Env.py)."""

    T: int = 10
    budget: int = 20
    p_seed: float = 1.0  # IM/Env.py always activates a selected seed
    n_paths: int = 400  # default Monte-Carlo width for evaluation


@dataclass
class RPPConfig:
    """Route/profit-planning environment defaults (RPP/Env.py)."""

    T: int = 10
    budget: int = 20
    max_daily_distance: float = 1.5
    distance_penalty: float = 5.0
    eta: float = 0.1  # cost eta * (k - 1)^2
    profit_factor: float = 0.05  # half-width of the daily uniform drift
    profit_min: float = 0.3
    start_node: int = 0
    n_paths: int = 400


@dataclass
class LogConfig:
    """Behaviour-log collection defaults for both benchmarks."""

    T: int = 10
    budget: int = 20
    n_episodes: int = 200
    seed: int = 0
    im_policies: Tuple[str, ...] = ("random", "score", "indeg", "outdeg")
    rpp_policies: Tuple[str, ...] = ("greedy", "nearest", "random")


@dataclass
class LawConfig:
    """L1 hypothesis-class defaults (filled in from M1 onwards)."""

    level: int = 3
    eps: float = 1.0  # LL >= LL_max - eps is kept in H_eps
    cap: int = 64  # maximum size of the hypothesis set
    tau: float = 1e-3  # probability gap that counts as discriminating
    prior_llm_hook: bool = False


@dataclass
class AbstractionConfig:
    """L2 defaults."""

    k_top: int = 3  # number of communities kept as regions
    checker_tau: float = 0.8
    non_dominance_delta: float = 0.05
    compile_samples: int = 64


@dataclass
class PlanningConfig:
    """L3 defaults."""

    mode: str = "mean"  # mean | cvar | maxmin | optimistic
    alpha: float = 0.5
    n_sims: int = 2000
    c_uct: float = 1.4
    widening: float = 1.0
    rollout_m: int = 64
    discount: float = 0.99


@dataclass
class DiagnosisConfig:
    """M4 defaults."""

    pit_alpha: float = 0.05
    decision_gap_tau: float = 0.5
    decision_gap_delta: float = 0.05


@dataclass
class StrataConfig:
    """Top-level configuration object."""

    seed: int = 0
    im: IMConfig = field(default_factory=IMConfig)
    rpp: RPPConfig = field(default_factory=RPPConfig)
    logs: LogConfig = field(default_factory=LogConfig)
    laws: LawConfig = field(default_factory=LawConfig)
    abstraction: AbstractionConfig = field(default_factory=AbstractionConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# Convenience alias used in the plan and in later milestones.
Config = StrataConfig
