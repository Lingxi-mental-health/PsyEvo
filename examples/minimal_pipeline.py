"""Minimal end-to-end use of the method-only PsyEvo interfaces."""

import numpy as np

from psyevo import (
    CrossClientPreferenceBuffer,
    HBSPConfig,
    HierarchicalBayesianSkillPolicy,
    PairJudgment,
    consistent_pair_graph,
    discounted_returns,
    ordinal_credit,
    population_graph_lipo_objective,
    select_candidate,
)


def local_graph(group_id: str, prefer_first: bool):
    support = 0.9 if prefer_first else 0.1
    return consistent_pair_graph(
        pointwise_scores=[0.7, 0.4],
        pair_judgments=[PairJudgment(0, 1, support, support)],
        group_id=group_id,
    )


def main() -> None:
    shared_q = np.array([[0.20, 0.45], [0.30, 0.25]])
    hbsp = HierarchicalBayesianSkillPolicy(
        shared_q,
        HBSPConfig(seed=7, thompson_sampling=True),
    )
    decision = hbsp.select_skill("client-a", state=0)

    graph_a = local_graph("session-a-turn-1", prefer_first=True)
    graph_b = local_graph("session-b-turn-1", prefer_first=False)
    delivered_a = select_candidate(graph_a)

    buffer = CrossClientPreferenceBuffer()
    buffer.add_graph(
        client_id="client-a", modality="modality-1", policy_version=0, graph=graph_a
    )
    buffer.add_graph(
        client_id="client-b", modality="modality-2", policy_version=0, graph=graph_b
    )
    batch = buffer.take_batch(
        min_groups=2,
        max_groups=2,
        min_clients=2,
        required_modalities=["modality-1", "modality-2"],
        max_groups_per_client=1,
    )

    policy_logp = {
        ("client-a", "session-a-turn-1"): [-1.0, -1.8],
        ("client-b", "session-b-turn-1"): [-1.2, -0.9],
    }
    reference_logp = {key: [-1.5, -1.5] for key in policy_logp}
    population_result = population_graph_lipo_objective(
        batch=batch,
        policy_logp=policy_logp,
        reference_logp=reference_logp,
        beta=0.05,
    )

    trajectory_credit = ordinal_credit([0.35, 0.80, 0.55])
    returns = discounted_returns(
        trajectory_credit.credits,
        terminal_reward=0.70,
        gamma=0.9,
    )
    posterior_update = hbsp.update(
        "client-a",
        state=0,
        action=decision.action,
        trajectory_return=returns[0],
    )

    print(f"delivered_candidate={delivered_a}")
    print(f"population_loss={population_result.loss:.6f}")
    print(f"posterior_mean={posterior_update.posterior_mean:.6f}")


if __name__ == "__main__":
    main()
