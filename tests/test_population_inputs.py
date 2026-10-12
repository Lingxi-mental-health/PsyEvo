"""Reject incomplete or ambiguous candidate groups before evaluating a loss."""

import unittest

from psyevo import (
    CrossClientBatch,
    CrossClientPreferenceBuffer,
    PairJudgment,
    consistent_pair_graph,
    population_graph_lipo_objective,
    torch_population_graph_lipo_loss,
)


class PopulationInputTests(unittest.TestCase):
    def setUp(self):
        # Candidate 2 abstains on both comparisons, so edge bounds alone cannot
        # detect its missing log probability.
        graph = consistent_pair_graph(
            [0.8, 0.5, 0.2],
            [PairJudgment(0, 1, 0.9, 0.9), PairJudgment(0, 2, 0.5, 0.5),
             PairJudgment(1, 2, 0.5, 0.5)],
            group_id="fixture",
        )
        buffer = CrossClientPreferenceBuffer()
        buffer.add_graph(client_id="client", modality="fixture", policy_version=0,
                         graph=graph)
        self.batch = buffer.take_batch(max_groups=1, min_clients=1)
        self.key = self.batch.groups[0].key

    def test_numpy_requires_exact_candidate_count(self):
        for count in (2, 4):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "candidate count"):
                population_graph_lipo_objective(
                    batch=self.batch, policy_logp={self.key: [-1.] * count},
                    reference_logp={self.key: [-2.] * count}, beta=0.05,
                )

    def test_duplicate_groups_rejected_by_both_entry_points(self):
        duplicate = CrossClientBatch(self.batch.groups * 2, {"client": 2},
                                     {"fixture": 2}, 2)
        for objective in (population_graph_lipo_objective, torch_population_graph_lipo_loss):
            with self.subTest(objective=objective.__name__), self.assertRaisesRegex(ValueError, "duplicate"):
                objective(batch=duplicate, policy_logp={self.key: [-1.] * 3},
                          reference_logp={self.key: [-2.] * 3}, beta=0.05)

    def test_torch_requires_exact_candidate_count(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch dependency is not installed")
        for count in (2, 4):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "candidate count"):
                torch_population_graph_lipo_loss(
                    batch=self.batch, policy_logp={self.key: torch.zeros(count)},
                    reference_logp={self.key: torch.zeros(count)}, beta=0.05,
                )
