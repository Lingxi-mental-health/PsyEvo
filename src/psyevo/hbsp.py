"""Hierarchical Bayesian Skill Policy (HBSP).

HBSP combines a frozen shared skill table with a client-private Normal
posterior. The implementation contains no domain-specific state encoder or
skill library; callers map their states and skills to integer indices.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HBSPConfig:
    """Hyperparameters for posterior inference and action selection."""

    prior_variance: float = 0.25
    observation_variance: float = 1.0
    temperature: float = 1.0
    epsilon: float = 0.05
    thompson_sampling: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.prior_variance) or self.prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive")
        if not np.isfinite(self.observation_variance) or self.observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")


@dataclass(frozen=True)
class SkillSelection:
    """Auditable output of one HBSP decision."""

    client_id: str
    state: int
    action: int
    probabilities: np.ndarray
    posterior_mean: np.ndarray
    posterior_sd: np.ndarray
    posterior_draw: np.ndarray


@dataclass(frozen=True)
class PosteriorUpdate:
    """Before/after values for one conjugate posterior update."""

    client_id: str
    state: int
    action: int
    residual: float
    prior_mean: float
    prior_precision: float
    posterior_mean: float
    posterior_precision: float


class HierarchicalBayesianSkillPolicy:
    """Client-private Normal posteriors over state-skill residual values.

    ``shared_values`` is the frozen table Q(s, a). Each new client receives an
    independent zero-mean residual prior, so one client's observations never
    modify another client's posterior.
    """

    def __init__(
        self,
        shared_values: np.ndarray,
        config: HBSPConfig | None = None,
    ) -> None:
        values = np.asarray(shared_values, dtype=float)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("shared_values must have shape [num_states, num_actions]")
        if not np.all(np.isfinite(values)):
            raise ValueError("shared_values must be finite")

        self.config = config or HBSPConfig()
        self._shared_values = values.copy()
        self._shared_values.setflags(write=False)
        self._rng = np.random.default_rng(self.config.seed)
        self._clients: dict[str, dict[str, np.ndarray]] = {}

    @property
    def shape(self) -> tuple[int, int]:
        return self._shared_values.shape

    @property
    def shared_values(self) -> np.ndarray:
        return self._shared_values.copy()

    @property
    def prior_precision(self) -> float:
        return 1.0 / self.config.prior_variance

    def _validate_state(self, state: int) -> int:
        state = int(state)
        if not 0 <= state < self.shape[0]:
            raise IndexError(f"state {state} is outside [0, {self.shape[0]})")
        return state

    def _validate_action(self, action: int) -> int:
        action = int(action)
        if not 0 <= action < self.shape[1]:
            raise IndexError(f"action {action} is outside [0, {self.shape[1]})")
        return action

    def _ensure_client(self, client_id: str) -> dict[str, np.ndarray]:
        client_id = str(client_id)
        if not client_id:
            raise ValueError("client_id must be non-empty")
        if client_id not in self._clients:
            self._clients[client_id] = {
                "mean": np.zeros(self.shape, dtype=float),
                "precision": np.full(self.shape, self.prior_precision, dtype=float),
            }
        return self._clients[client_id]

    def posterior_mean(self, client_id: str, state: int) -> np.ndarray:
        state = self._validate_state(state)
        return self._ensure_client(client_id)["mean"][state].copy()

    def posterior_sd(self, client_id: str, state: int) -> np.ndarray:
        state = self._validate_state(state)
        precision = self._ensure_client(client_id)["precision"][state]
        return np.sqrt(1.0 / precision)

    def _valid_actions(self, valid_actions: Iterable[int] | None) -> np.ndarray:
        if valid_actions is None:
            return np.arange(self.shape[1], dtype=int)
        actions = np.asarray(
            [self._validate_action(a) for a in valid_actions], dtype=int
        )
        if actions.size == 0:
            raise ValueError("valid_actions must contain at least one action")
        if np.unique(actions).size != actions.size:
            raise ValueError("valid_actions must not contain duplicates")
        return actions

    def action_probabilities(
        self,
        client_id: str,
        state: int,
        valid_actions: Iterable[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return behavior probabilities and posterior decision evidence.

        The returned tuple is ``(probabilities, mean, sd, draw)``. Invalid
        actions receive probability zero.
        """

        state = self._validate_state(state)
        actions = self._valid_actions(valid_actions)
        client = self._ensure_client(client_id)
        mean = client["mean"][state].copy()
        sd = np.sqrt(1.0 / client["precision"][state])
        draw = (
            mean + sd * self._rng.standard_normal(mean.shape)
            if self.config.thompson_sampling
            else mean.copy()
        )

        logits = self.config.temperature * (
            self._shared_values[state, actions] + draw[actions]
        )
        logits -= np.max(logits)
        softmax = np.exp(logits)
        softmax /= np.sum(softmax)
        behavior = (1.0 - self.config.epsilon) * softmax
        behavior += self.config.epsilon / actions.size

        probabilities = np.zeros(self.shape[1], dtype=float)
        probabilities[actions] = behavior
        return probabilities, mean, sd, draw

    def select_skill(
        self,
        client_id: str,
        state: int,
        valid_actions: Iterable[int] | None = None,
    ) -> SkillSelection:
        probabilities, mean, sd, draw = self.action_probabilities(
            client_id, state, valid_actions
        )
        action = int(self._rng.choice(self.shape[1], p=probabilities))
        return SkillSelection(
            client_id=str(client_id),
            state=int(state),
            action=action,
            probabilities=probabilities,
            posterior_mean=mean,
            posterior_sd=sd,
            posterior_draw=draw,
        )

    def update(
        self,
        client_id: str,
        state: int,
        action: int,
        trajectory_return: float,
    ) -> PosteriorUpdate:
        """Update one posterior cell using d_t = G_t - Q(s_t, a_t)."""

        state = self._validate_state(state)
        action = self._validate_action(action)
        trajectory_return = float(trajectory_return)
        if not np.isfinite(trajectory_return):
            raise ValueError("trajectory_return must be finite")

        client = self._ensure_client(client_id)
        mean = float(client["mean"][state, action])
        precision = float(client["precision"][state, action])
        residual = trajectory_return - float(self._shared_values[state, action])
        observation_precision = 1.0 / self.config.observation_variance
        new_precision = precision + observation_precision
        new_mean = (precision * mean + observation_precision * residual) / new_precision
        client["mean"][state, action] = new_mean
        client["precision"][state, action] = new_precision

        return PosteriorUpdate(
            client_id=str(client_id),
            state=state,
            action=action,
            residual=residual,
            prior_mean=mean,
            prior_precision=precision,
            posterior_mean=new_mean,
            posterior_precision=new_precision,
        )

    def state_dict(self) -> dict:
        """Return a JSON-serializable snapshot of private posterior state."""

        return {
            "shape": list(self.shape),
            "config": {
                "prior_variance": self.config.prior_variance,
                "observation_variance": self.config.observation_variance,
                "temperature": self.config.temperature,
                "epsilon": self.config.epsilon,
                "thompson_sampling": self.config.thompson_sampling,
                "seed": self.config.seed,
            },
            "clients": {
                client_id: {
                    "mean": state["mean"].tolist(),
                    "precision": state["precision"].tolist(),
                }
                for client_id, state in self._clients.items()
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore posteriors after validating their shape and precision."""

        if tuple(state_dict.get("shape", ())) != self.shape:
            raise ValueError("state_dict shape does not match shared_values")
        restored: dict[str, dict[str, np.ndarray]] = {}
        for client_id, state in state_dict.get("clients", {}).items():
            mean = np.asarray(state["mean"], dtype=float)
            precision = np.asarray(state["precision"], dtype=float)
            if mean.shape != self.shape or precision.shape != self.shape:
                raise ValueError(f"invalid posterior shape for client {client_id!r}")
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(precision)):
                raise ValueError(f"non-finite posterior for client {client_id!r}")
            if np.any(precision < self.prior_precision):
                raise ValueError(
                    f"posterior precision decreased for client {client_id!r}"
                )
            restored[str(client_id)] = {
                "mean": mean.copy(),
                "precision": precision.copy(),
            }
        self._clients = restored
