"""State-conditioned Ordinal Credit Assignment (SOCA).

The module has two independent interfaces:

1. SOCA turns same-state candidate comparisons into explicit preference
   edges for LiPO.
2. Ordinal projection turns realized trajectory outcomes into HBSP credit.

Keeping these interfaces separate prevents unshown candidates from receiving
fabricated client outcomes.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PairJudgment:
    """Two-order support for original candidate ``i`` in an unordered pair."""

    i: int
    j: int
    forward_i_share: float
    reverse_i_share: float


@dataclass(frozen=True)
class PreferenceEdge:
    """A supported directed preference used by LiPO."""

    winner: int
    loser: int
    weight: float


@dataclass(frozen=True)
class SOCAGraph:
    """Acyclic weighted preference graph for one candidate group."""

    group_id: str
    pointwise_scores: tuple[float, ...]
    selected_edges: tuple[PreferenceEdge, ...]
    depths: tuple[int, ...]
    stable_edge_count: int
    cycle_skip_count: int


@dataclass(frozen=True)
class OrdinalCredit:
    """Tie-aware ordinal projection of realized trajectory outcomes."""

    raw_scores: tuple[float, ...]
    midranks: tuple[float, ...]
    tiers: tuple[int, ...]
    levels: tuple[float, ...]
    credits: tuple[float, ...]
    mean: float
    buckets: int
    spread: float
    tie_count: int


def _finite_values(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _bounded_mean_projection(
    values: Sequence[float], target_mean: float
) -> list[float]:
    """Apply one shared shift, clip to [0, 1], and preserve the mean."""

    values = [float(value) for value in values]
    target = min(1.0, max(0.0, float(target_mean)))
    lower = -max(values) - 1.0
    upper = 1.0 - min(values) + 1.0
    for _ in range(80):
        shift = (lower + upper) / 2.0
        projected_mean = sum(
            min(1.0, max(0.0, value + shift)) for value in values
        ) / len(values)
        if projected_mean < target:
            lower = shift
        else:
            upper = shift
    shift = (lower + upper) / 2.0
    return [min(1.0, max(0.0, value + shift)) for value in values]


def ordinal_credit(
    scores: Sequence[float],
    *,
    buckets: int = 3,
    spread: float = 0.5,
) -> OrdinalCredit:
    """Project realized scores onto tie-aware ordinal levels.

    A shared pre-clipping shift preserves the original group mean. Exact ties
    receive the same midrank and therefore the same projected credit.
    """

    raw = _finite_values(scores, "scores")
    if any(not 0.0 <= score <= 1.0 for score in raw):
        raise ValueError("scores must lie in [0, 1]")
    if not math.isfinite(spread) or spread < 0.0:
        raise ValueError("spread must be finite and non-negative")

    n = len(raw)
    bucket_count = 1 if n == 1 else max(2, min(int(buckets), n))
    order = sorted(range(n), key=raw.__getitem__)
    midranks = [0.0] * n
    start = 0
    while start < n:
        end = start + 1
        while end < n and raw[order[end]] == raw[order[start]]:
            end += 1
        midrank = (start + end - 1) / 2.0
        for position in range(start, end):
            midranks[order[position]] = midrank
        start = end

    if n == 1:
        tiers = [0]
        levels = [0.0]
    else:
        tiers = [
            min(bucket_count - 1, int(midrank * bucket_count // n))
            for midrank in midranks
        ]
        levels = [2.0 * tier / (bucket_count - 1) - 1.0 for tier in tiers]

    mean = sum(raw) / n
    credits = _bounded_mean_projection(
        [mean + level * float(spread) for level in levels], mean
    )
    return OrdinalCredit(
        raw_scores=tuple(raw),
        midranks=tuple(midranks),
        tiers=tuple(tiers),
        levels=tuple(levels),
        credits=tuple(credits),
        mean=mean,
        buckets=bucket_count,
        spread=float(spread),
        tie_count=n - len(set(raw)),
    )


def discounted_returns(
    credits: Iterable[float],
    *,
    terminal_reward: float,
    gamma: float = 0.9,
) -> tuple[float, ...]:
    """Compute G_t = credit_t + gamma * G_(t+1) backwards in time."""

    credit_values = _finite_values(list(credits), "credits")
    terminal_reward = float(terminal_reward)
    gamma = float(gamma)
    if not math.isfinite(terminal_reward):
        raise ValueError("terminal_reward must be finite")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    running = terminal_reward
    returns = [0.0] * len(credit_values)
    for index in range(len(credit_values) - 1, -1, -1):
        running = credit_values[index] + gamma * running
        returns[index] = running
    return tuple(returns)


def _acyclic_after_add(
    candidate_count: int,
    edges: Sequence[PreferenceEdge],
    winner: int,
    loser: int,
) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(candidate_count)]
    for edge in edges:
        adjacency[edge.winner].append(edge.loser)
    stack = [loser]
    visited = set()
    while stack:
        node = stack.pop()
        if node == winner:
            return False
        if node not in visited:
            visited.add(node)
            stack.extend(adjacency[node])
    return True


def _graph_depths(
    candidate_count: int, edges: Sequence[PreferenceEdge]
) -> tuple[int, ...]:
    adjacency: list[list[int]] = [[] for _ in range(candidate_count)]
    indegree = [0] * candidate_count
    for edge in edges:
        adjacency[edge.winner].append(edge.loser)
        indegree[edge.loser] += 1

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    topological_order: list[int] = []
    while ready:
        node = ready.pop()
        topological_order.append(node)
        for loser in adjacency[node]:
            indegree[loser] -= 1
            if indegree[loser] == 0:
                ready.append(loser)
    if len(topological_order) != candidate_count:
        raise ValueError("preference graph must be acyclic")

    depths = [0] * candidate_count
    for winner in reversed(topological_order):
        if adjacency[winner]:
            depths[winner] = 1 + max(depths[loser] for loser in adjacency[winner])
    return tuple(depths)


def consistent_pair_graph(
    pointwise_scores: Sequence[float],
    pair_judgments: Sequence[PairJudgment],
    *,
    group_id: str,
    min_strength: float = 0.0,
) -> SOCAGraph:
    """Construct the SOCA graph for one candidate state.

    Pairwise judgments alone decide which directed edges are eligible.
    Pointwise scores only break display ties; they never create, remove, or
    reverse an edge.
    """

    scores = _finite_values(pointwise_scores, "pointwise_scores")
    if len(scores) < 2:
        raise ValueError("SOCA requires at least two candidates")
    if not group_id:
        raise ValueError("group_id must be non-empty")
    if not 0.0 <= min_strength <= 1.0:
        raise ValueError("min_strength must lie in [0, 1]")

    candidate_count = len(scores)
    expected_pairs = set(itertools.combinations(range(candidate_count), 2))
    seen_pairs = set()
    stable: list[tuple[float, int, int]] = []

    for judgment in pair_judgments:
        i, j = int(judgment.i), int(judgment.j)
        if not (0 <= i < j < candidate_count) or (i, j) in seen_pairs:
            raise ValueError("pair identities must be unique canonical indices i < j")
        seen_pairs.add((i, j))
        forward = float(judgment.forward_i_share)
        reverse = float(judgment.reverse_i_share)
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (forward, reverse)
        ):
            raise ValueError("pairwise support shares must be probabilities")

        if forward > 0.5 and reverse > 0.5:
            winner, loser = i, j
            strength = 2.0 * min(forward, reverse) - 1.0
        elif forward < 0.5 and reverse < 0.5:
            winner, loser = j, i
            strength = 2.0 * min(1.0 - forward, 1.0 - reverse) - 1.0
        else:
            continue
        if strength <= 0.0 or strength < min_strength:
            continue
        stable.append((strength, winner, loser))

    if seen_pairs != expected_pairs:
        raise ValueError("one two-order judgment is required for every candidate pair")

    stable.sort(
        key=lambda item: (
            -item[0],
            hashlib.sha256(
                f"{group_id}:{min(item[1:])}:{max(item[1:])}".encode()
            ).hexdigest(),
        )
    )
    selected: list[PreferenceEdge] = []
    cycle_skips = 0
    for strength, winner, loser in stable:
        if _acyclic_after_add(candidate_count, selected, winner, loser):
            selected.append(PreferenceEdge(winner, loser, strength))
        else:
            cycle_skips += 1

    return SOCAGraph(
        group_id=str(group_id),
        pointwise_scores=tuple(scores),
        selected_edges=tuple(selected),
        depths=_graph_depths(candidate_count, selected),
        stable_edge_count=len(stable),
        cycle_skip_count=cycle_skips,
    )


def select_candidate(graph: SOCAGraph) -> int:
    """Select maximum depth, then pointwise score, then lowest index."""

    return max(
        range(len(graph.pointwise_scores)),
        key=lambda index: (
            graph.depths[index],
            graph.pointwise_scores[index],
            -index,
        ),
    )
