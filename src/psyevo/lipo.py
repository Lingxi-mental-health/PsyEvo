"""Weighted LiPO objective over explicit SOCA preference edges."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .population import CrossClientBatch
from .soca import PreferenceEdge


@dataclass(frozen=True)
class PopulationLiPOResult:
    """Loss and per-group gradients for one shared-adapter update."""

    loss: float
    group_losses: dict[tuple[str, str], float]
    utility_gradients: dict[tuple[str, str], np.ndarray]
    client_distribution: dict[str, int]
    modality_distribution: dict[str, int]


def policy_reference_utilities(
    policy_logp: Sequence[float],
    reference_logp: Sequence[float],
    *,
    beta: float,
) -> np.ndarray:
    """Compute h_i = beta * (log pi_i - log pi_ref_i)."""

    policy = np.asarray(policy_logp, dtype=float)
    reference = np.asarray(reference_logp, dtype=float)
    beta = float(beta)
    if policy.ndim != 1 or policy.shape != reference.shape or policy.size < 2:
        raise ValueError("policy_logp and reference_logp must be equal 1-D arrays")
    if not np.all(np.isfinite(policy)) or not np.all(np.isfinite(reference)):
        raise ValueError("log probabilities must be finite")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    return beta * (policy - reference)


def _validated_edges(
    edges: Sequence[PreferenceEdge], candidate_count: int
) -> tuple[PreferenceEdge, ...]:
    if not edges:
        raise ValueError("LiPO requires at least one explicit preference edge")
    checked = []
    seen = set()
    for edge in edges:
        winner, loser, weight = int(edge.winner), int(edge.loser), float(edge.weight)
        if not 0 <= winner < candidate_count or not 0 <= loser < candidate_count:
            raise ValueError("edge index is outside the candidate group")
        if winner == loser or (winner, loser) in seen:
            raise ValueError("edges must be distinct and connect different candidates")
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise ValueError("edge weights must lie in (0, 1]")
        seen.add((winner, loser))
        checked.append(PreferenceEdge(winner, loser, weight))
    return tuple(checked)


def graph_lipo_objective(
    *,
    policy_logp: Sequence[float],
    reference_logp: Sequence[float],
    edges: Sequence[PreferenceEdge],
    beta: float,
) -> tuple[float, np.ndarray]:
    """Return selected-edge-normalized LiPO loss and d(loss)/d(utility).

    The analytic utility gradient supports memory-efficient two-pass training:
    first compute detached utilities, then replay one policy log-probability at
    a time with coefficient ``beta * utility_gradient[i]``.
    """

    utilities = policy_reference_utilities(policy_logp, reference_logp, beta=beta)
    checked_edges = _validated_edges(edges, utilities.size)
    gradient = np.zeros_like(utilities)
    loss = 0.0
    normalization = float(len(checked_edges))

    for edge in checked_edges:
        margin = utilities[edge.winner] - utilities[edge.loser]
        loss += edge.weight * np.logaddexp(0.0, -margin)
        if margin >= 0.0:
            exp_negative = np.exp(-margin)
            sigmoid_negative_margin = exp_negative / (1.0 + exp_negative)
        else:
            sigmoid_negative_margin = 1.0 / (1.0 + np.exp(margin))
        coefficient = edge.weight * sigmoid_negative_margin / normalization
        gradient[edge.winner] -= coefficient
        gradient[edge.loser] += coefficient

    return float(loss / normalization), gradient


def _validate_population_inputs(batch, policy_logp, reference_logp) -> None:
    if not batch.groups:
        raise ValueError("population LiPO requires a non-empty batch")
    expected = {group.key for group in batch.groups}
    if len(expected) != len(batch.groups):
        raise ValueError("duplicate candidate groups in population batch")
    if set(policy_logp) != expected or set(reference_logp) != expected:
        raise ValueError("policy/reference log-probability keys must match the batch")
    for group in batch.groups:
        candidate_count = len(group.graph.pointwise_scores)
        for values in (policy_logp[group.key], reference_logp[group.key]):
            try:
                count = len(values)
            except TypeError as exc:
                raise ValueError("log probabilities must be candidate sequences") from exc
            if count != candidate_count:
                raise ValueError("log-probability candidate count must match the SOCA graph")


def population_graph_lipo_objective(
    *,
    batch: CrossClientBatch,
    policy_logp: dict[tuple[str, str], Sequence[float]],
    reference_logp: dict[tuple[str, str], Sequence[float]],
    beta: float,
) -> PopulationLiPOResult:
    """Evaluate shared-adapter LiPO with equal weight per candidate group.

    Each graph is normalized by its selected edge count, then graph losses
    are averaged over the batch. Returned gradients include the batch mean
    factor; client identity does not introduce an additional loss weight.
    """

    _validate_population_inputs(batch, policy_logp, reference_logp)
    group_count = len(batch.groups)
    losses: dict[tuple[str, str], float] = {}
    gradients: dict[tuple[str, str], np.ndarray] = {}
    total = 0.0
    for group in batch.groups:
        loss, gradient = graph_lipo_objective(
            policy_logp=policy_logp[group.key],
            reference_logp=reference_logp[group.key],
            edges=group.graph.selected_edges,
            beta=beta,
        )
        scale = 1.0 / group_count
        losses[group.key] = loss
        gradients[group.key] = gradient * scale
        total += loss * scale
    return PopulationLiPOResult(
        loss=float(total),
        group_losses=losses,
        utility_gradients=gradients,
        client_distribution=dict(batch.client_distribution),
        modality_distribution=dict(batch.modality_distribution),
    )


def torch_graph_lipo_loss(
    policy_logp,
    reference_logp,
    edges: Sequence[PreferenceEdge],
    *,
    beta: float,
):
    """PyTorch/autograd form of the same LiPO objective.

    PyTorch is imported lazily so the reference package remains lightweight.
    ``reference_logp`` is computed with the shared adapter disabled.
    """

    try:
        import torch
        from torch.nn import functional
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "install the 'torch' extra to use torch_graph_lipo_loss"
        ) from exc

    if policy_logp.ndim != 1 or policy_logp.shape != reference_logp.shape:
        raise ValueError("policy_logp and reference_logp must be equal 1-D tensors")
    if policy_logp.numel() < 2:
        raise ValueError("at least two candidates are required")
    if not torch.isfinite(policy_logp).all() or not torch.isfinite(reference_logp).all():
        raise ValueError("log probabilities must be finite")
    if not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be positive and finite")
    checked_edges = _validated_edges(edges, int(policy_logp.numel()))
    utilities = float(beta) * (policy_logp - reference_logp.detach())
    terms = [
        edge.weight
        * functional.softplus(-(utilities[edge.winner] - utilities[edge.loser]))
        for edge in checked_edges
    ]
    return torch.stack(terms).sum() / len(terms)


def torch_population_graph_lipo_loss(
    *,
    batch: CrossClientBatch,
    policy_logp: dict[tuple[str, str], object],
    reference_logp: dict[tuple[str, str], object],
    beta: float,
):
    """Autograd population loss for updating one shared adapter."""

    _validate_population_inputs(batch, policy_logp, reference_logp)
    group_count = len(batch.groups)
    terms = []
    for group in batch.groups:
        group_loss = torch_graph_lipo_loss(
            policy_logp[group.key],
            reference_logp[group.key],
            group.graph.selected_edges,
            beta=beta,
        )
        terms.append(group_loss / group_count)
    return sum(terms)
