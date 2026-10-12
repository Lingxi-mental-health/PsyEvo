"""Cross-client population batching for SOCA preference graphs.

SOCA comparisons remain local to one client and one state.  This module only
pools the resulting, auditable graphs for a shared LiPO adapter update.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

from .soca import SOCAGraph


@dataclass(frozen=True)
class PopulationPreferenceGroup:
    """One client-local SOCA graph admitted to population training."""

    client_id: str
    modality: str
    policy_version: int
    graph: SOCAGraph

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise ValueError("client_id must be non-empty")
        if not self.modality.strip():
            raise ValueError("modality must be non-empty")
        if int(self.policy_version) < 0:
            raise ValueError("policy_version must be non-negative")
        if not self.graph.selected_edges:
            raise ValueError("population training requires at least one SOCA edge")

    @property
    def key(self) -> tuple[str, str]:
        return self.client_id, self.graph.group_id


@dataclass(frozen=True)
class CrossClientBatch:
    """An immutable, auditable shared-adapter training batch."""

    groups: tuple[PopulationPreferenceGroup, ...]
    client_distribution: dict[str, int]
    modality_distribution: dict[str, int]
    edge_count: int


class CrossClientPreferenceBuffer:
    """Pool client-local SOCA graphs without constructing cross-client edges.

    Selection is deterministic and round-robin over clients. Required
    modalities and a per-client cap can be used to prevent a long client from
    dominating a population update. Failed batch construction is atomic: the
    buffer is unchanged.
    """

    def __init__(self) -> None:
        self._groups: dict[tuple[str, str], PopulationPreferenceGroup] = {}
        self._sequence: dict[tuple[str, str], int] = {}
        self._next_sequence = 0

    def __len__(self) -> int:
        return len(self._groups)

    @property
    def edge_count(self) -> int:
        return sum(len(group.graph.selected_edges) for group in self._groups.values())

    def add_graph(
        self,
        *,
        client_id: str,
        modality: str,
        policy_version: int,
        graph: SOCAGraph,
    ) -> PopulationPreferenceGroup:
        group = PopulationPreferenceGroup(
            str(client_id), str(modality), int(policy_version), graph
        )
        if group.key in self._groups:
            raise ValueError(f"duplicate client/group key: {group.key!r}")
        self._groups[group.key] = group
        self._sequence[group.key] = self._next_sequence
        self._next_sequence += 1
        return group

    def extend(self, groups: Iterable[PopulationPreferenceGroup]) -> None:
        groups = tuple(groups)
        keys = [group.key for group in groups]
        if len(keys) != len(set(keys)):
            raise ValueError("groups must have distinct client/group keys")
        conflicts = [key for key in keys if key in self._groups]
        if conflicts:
            raise ValueError(f"duplicate client/group key: {conflicts[0]!r}")
        for group in groups:
            self.add_graph(
                client_id=group.client_id,
                modality=group.modality,
                policy_version=group.policy_version,
                graph=group.graph,
            )

    def distributions(self) -> tuple[dict[str, int], dict[str, int]]:
        return (
            dict(Counter(group.client_id for group in self._groups.values())),
            dict(Counter(group.modality for group in self._groups.values())),
        )

    def take_batch(
        self,
        *,
        max_groups: int,
        min_groups: int = 1,
        min_clients: int = 2,
        required_modalities: Iterable[str] = (),
        max_groups_per_client: int | None = None,
    ) -> CrossClientBatch:
        """Remove and return one balanced population batch atomically."""

        max_groups = int(max_groups)
        min_groups = int(min_groups)
        min_clients = int(min_clients)
        if min_groups < 1 or max_groups < min_groups:
            raise ValueError("require 1 <= min_groups <= max_groups")
        if min_clients < 1:
            raise ValueError("min_clients must be positive")
        if max_groups_per_client is not None and int(max_groups_per_client) < 1:
            raise ValueError("max_groups_per_client must be positive")

        required = tuple(dict.fromkeys(str(item) for item in required_modalities))
        if any(not modality.strip() for modality in required):
            raise ValueError("required_modalities must be non-empty strings")
        if len(required) > max_groups:
            raise ValueError("required modality count cannot exceed max_groups")
        ordered = sorted(self._groups.values(), key=lambda g: self._sequence[g.key])
        by_client: dict[str, deque[PopulationPreferenceGroup]] = defaultdict(deque)
        for group in ordered:
            by_client[group.client_id].append(group)

        selected: list[PopulationPreferenceGroup] = []
        selected_keys: set[tuple[str, str]] = set()
        client_counts: Counter[str] = Counter()

        def eligible(group: PopulationPreferenceGroup) -> bool:
            if group.key in selected_keys:
                return False
            return max_groups_per_client is None or client_counts[
                group.client_id
            ] < int(max_groups_per_client)

        def choose(group: PopulationPreferenceGroup) -> None:
            selected.append(group)
            selected_keys.add(group.key)
            client_counts[group.client_id] += 1

        # Reserve coverage for every required modality first.
        for modality in required:
            match = next(
                (
                    group
                    for group in ordered
                    if group.modality == modality and eligible(group)
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"required modality has no eligible group: {modality!r}"
                )
            choose(match)

        # Add unseen clients before taking a second group from any client.
        for client_id in sorted(by_client):
            if len(client_counts) >= min_clients or len(selected) >= max_groups:
                break
            if client_id in client_counts:
                continue
            match = next(
                (group for group in by_client[client_id] if eligible(group)),
                None,
            )
            if match is not None:
                choose(match)

        # Deterministic round-robin fill.
        while len(selected) < max_groups:
            added = False
            for client_id in sorted(by_client):
                match = next(
                    (group for group in by_client[client_id] if eligible(group)),
                    None,
                )
                if match is not None:
                    choose(match)
                    added = True
                    if len(selected) == max_groups:
                        break
            if not added:
                break

        if len(selected) < min_groups:
            raise ValueError(
                f"only {len(selected)} eligible groups; require {min_groups}"
            )
        if len(client_counts) < min_clients:
            raise ValueError(
                f"only {len(client_counts)} clients; require {min_clients}"
            )

        for group in selected:
            del self._groups[group.key]
            del self._sequence[group.key]
        modality_counts = Counter(group.modality for group in selected)
        return CrossClientBatch(
            groups=tuple(selected),
            client_distribution=dict(client_counts),
            modality_distribution=dict(modality_counts),
            edge_count=sum(len(group.graph.selected_edges) for group in selected),
        )

    def take_balanced_edge_batch(
        self, *, pairs: int = 320, min_groups: int = 32,
        modalities: Iterable[str] = ("BT", "CBT", "HET", "PDT", "PMT"),
    ) -> CrossClientBatch | None:
        """Reserve complete graphs with equal retained-edge quotas per modality.

        Deterministic subset selection maximizes the number of complete groups
        at each modality's edge budget. No graph is truncated; unavailable exact
        budgets return None without consuming data. This is a reference sampler,
        not a claim to reproduce the experimental runner's sampling order.
        """
        modalities = tuple(modalities)
        if (pairs < 1 or min_groups < 1 or not modalities
                or len(set(modalities)) != len(modalities)
                or pairs % len(modalities)):
            raise ValueError("require positive budgets and distinct equal-share modalities")
        quota = pairs // len(modalities)
        selected = []
        ordered = sorted(self._groups.values(), key=lambda g: self._sequence[g.key])
        for modality in modalities:
            queues = defaultdict(deque)
            for group in ordered:
                if group.modality == modality:
                    queues[group.client_id].append(group)
            candidates = []
            while any(queues.values()):
                for client in sorted(queues):
                    if queues[client]:
                        candidates.append(queues[client].popleft())
            best = {0: ()}
            for group in candidates:
                edges = len(group.graph.selected_edges)
                for count, subset in sorted(tuple(best.items()), reverse=True):
                    total = count + edges
                    if total <= quota and (total not in best or len(subset) + 1 > len(best[total])):
                        best[total] = subset + (group,)
            if quota not in best:
                return None
            selected.extend(best[quota])
        if len(selected) < min_groups:
            return None
        for group in selected:
            del self._groups[group.key]
            del self._sequence[group.key]
        return CrossClientBatch(
            tuple(selected), dict(Counter(g.client_id for g in selected)),
            dict(Counter(g.modality for g in selected)), pairs,
        )

    def return_batch(self, batch: CrossClientBatch) -> None:
        """Atomically return every group after a failed optimizer step."""

        self.extend(batch.groups)
