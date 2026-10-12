"""Method-level session scheduling; model execution is supplied by a backend.

One coordinator belongs to one population learning scope, not one client.
Its methods are serialized by a lock. Training receives an immutable adapter
reference and MUST produce a separate candidate without mutating the active
adapter. Existing sessions keep their original reference after acceptance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from threading import RLock
from typing import Callable

from .population import CrossClientBatch


@dataclass(frozen=True)
class PaperMethodConfig:
    """Paper configuration, not downloaded weights or a benchmark runner."""

    counselor: str = "Qwen3-32B"
    retrieval: str = "BGE-M3"
    simulated_client: str = "DeepSeek-V3.1"
    lora_rank: int = 8
    learning_rate: float = 1e-5
    beta: float = 0.05
    batch_size: int = 2
    gradient_accumulation: int = 4
    max_optimizer_steps: int = 3
    epochs: int = 1
    max_sequence_length: int = 1024
    shadow_responses: int = 4
    candidate_count: int = 5
    stable_edge_threshold: float = 0.1
    update_sessions: int = 5
    preference_pairs: int = 320
    min_groups: int = 32
    modalities: tuple[str, ...] = ("BT", "CBT", "HET", "PDT", "PMT")

    def __post_init__(self) -> None:
        counts = (self.lora_rank, self.batch_size, self.gradient_accumulation,
                  self.max_optimizer_steps, self.epochs, self.max_sequence_length,
                  self.update_sessions, self.preference_pairs, self.min_groups)
        if any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError("counts must be positive integers")
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be nonempty and distinct")
        if self.preference_pairs % len(self.modalities):
            raise ValueError("pair budget must allow equal modality shares")
        if self.shadow_responses < 0 or self.candidate_count != self.shadow_responses + 1:
            raise ValueError("candidate count includes the delivered response")
        if any(not math.isfinite(x) or x <= 0 for x in (self.beta, self.learning_rate)):
            raise ValueError("learning rate and beta must be positive")
        if not 0 <= self.stable_edge_threshold <= 1:
            raise ValueError("edge threshold must lie in [0, 1]")

    def check_batch(self, batch: CrossClientBatch) -> None:
        """Validate actual graph contents, not cached distribution metadata.

        Five modalities each at least 20% of exactly 320 retained edges means
        64 edges per modality. Graphs are never cut to meet this budget.
        """
        if len(batch.groups) < self.min_groups:
            raise ValueError("insufficient independent candidate groups")
        if len({g.key for g in batch.groups}) != len(batch.groups):
            raise ValueError("duplicate candidate groups")
        counts: Counter[str] = Counter()
        for group in batch.groups:
            if not group.graph.selected_edges:
                raise ValueError("empty preference graph")
            counts[group.modality] += len(group.graph.selected_edges)
        quota = self.preference_pairs // len(self.modalities)
        if dict(counts) != dict.fromkeys(self.modalities, quota):
            raise ValueError("batch must satisfy the exact pair budget and modality shares")


@dataclass(frozen=True)
class AdapterVersion:
    version: int
    checkpoint: str

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 0 or not self.checkpoint:
            raise ValueError("adapter requires a nonnegative version and checkpoint")


class PopulationSessionScheduler:
    """Five completed population sessions make an update eligible, not mandatory.

    HBSP/memory updates are performed by the caller on every completed session,
    independently of this shared-adapter gate. Session IDs must be unique within
    a client for the life of this coordinator. Restart starts a fresh scope;
    this implementation does not promise durable resume or remote publication.
    """

    def __init__(self, initial: AdapterVersion, config: PaperMethodConfig = PaperMethodConfig()):
        self.config = config
        self._active = initial
        self._sessions: dict[tuple[str, str], AdapterVersion] = {}
        self._completed: set[tuple[str, str]] = set()
        self._pending = 0
        self._lock = RLock()

    @property
    def active(self) -> AdapterVersion:
        with self._lock:
            return self._active

    @property
    def pending_sessions(self) -> int:
        with self._lock:
            return self._pending

    @property
    def eligible(self) -> bool:
        with self._lock:
            return self._pending >= self.config.update_sessions

    def begin_session(self, client_id: str, session_id: str) -> AdapterVersion:
        with self._lock:
            key = (client_id, session_id)
            if not client_id or not session_id or key in self._sessions or key in self._completed:
                raise ValueError("session identity must be nonempty and unique")
            self._sessions[key] = self._active
            return self._active

    def session_adapter(self, client_id: str, session_id: str) -> AdapterVersion:
        with self._lock:
            return self._sessions[(client_id, session_id)]

    def complete_session(self, client_id: str, session_id: str) -> bool:
        """Count exactly once; duplicate delivery does not advance the gate."""
        with self._lock:
            key = (client_id, session_id)
            if key in self._completed:
                return False
            if key not in self._sessions:
                raise ValueError("session was not started")
            del self._sessions[key]
            self._completed.add(key)
            self._pending += 1
            return True

    def try_update(
        self,
        *,
        reserve_batch: Callable[[], CrossClientBatch | None],
        return_batch: Callable[[CrossClientBatch], None],
        train_candidate: Callable[[CrossClientBatch, AdapterVersion, PaperMethodConfig], str],
        validate_candidate: Callable[[str], bool],
    ) -> bool:
        """Reserve, check, train in isolation, validate, then switch the reference.

        reserve_batch must be atomic and return None when no suitable batch is
        available. return_batch must restore that reservation atomically. No
        evidence is requested before the five-session gate. Exceptions restore
        evidence and propagate; rejection preserves the active adapter and gate.
        Physical weights, optimizer snapshots, serving and checkpoint retention
        remain backend responsibilities; this does not mutate or roll back them.
        """
        with self._lock:
            if not self.eligible:
                return False
            batch = reserve_batch()
            if batch is None:
                return False
            try:
                self.config.check_batch(batch)
                checkpoint = train_candidate(batch, self._active, self.config)
                if not checkpoint or checkpoint == self._active.checkpoint:
                    raise ValueError("training must create a separate candidate checkpoint")
                accepted = validate_candidate(checkpoint)
            except BaseException:
                # Restore reservations on cancellation too, then propagate it.
                # A restoration failure retains the original error as context.
                return_batch(batch)
                raise
            if not accepted:
                return_batch(batch)
                return False
            self._active = AdapterVersion(self._active.version + 1, checkpoint)
            self._pending = 0
            return True
