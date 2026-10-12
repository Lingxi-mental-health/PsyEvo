"""Boundary checks without changing the paper's objective or batch quotas."""

import unittest
from unittest.mock import Mock

from psyevo import (
    AdapterVersion, HBSPConfig, PaperMethodConfig, PopulationSessionScheduler,
    CrossClientPreferenceBuffer, PairJudgment, PreferenceEdge,
    consistent_pair_graph, ordinal_credit,
)


class RobustnessTests(unittest.TestCase):
    def test_nonfinite_hbsp_configuration(self):
        for field in ("prior_variance", "observation_variance", "temperature", "epsilon"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    HBSPConfig(**{field: value})

    def test_invalid_credit_spread(self):
        for value in (float("nan"), float("inf"), -float("inf"), -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ordinal_credit([0.2, 0.8], spread=value)
        self.assertEqual(ordinal_credit([0.2, 0.8], spread=0).credits, (0.5, 0.5))

    def make_update(self):
        config = PaperMethodConfig(preference_pairs=1, min_groups=1,
                                   modalities=("CBT",), update_sessions=1)
        scheduler = PopulationSessionScheduler(AdapterVersion(0, "initial"), config)
        scheduler.begin_session("client", "session")
        scheduler.complete_session("client", "session")
        buffer = CrossClientPreferenceBuffer()
        graph = consistent_pair_graph([0.7, 0.4], [PairJudgment(0, 1, 0.9, 0.9)],
                                      group_id="group")
        buffer.add_graph(client_id="client", modality="CBT", policy_version=0, graph=graph)
        callbacks = dict(
            reserve_batch=lambda: buffer.take_balanced_edge_batch(
                pairs=1, min_groups=1, modalities=("CBT",)),
            return_batch=buffer.return_batch,
            train_candidate=lambda *args: "candidate",
            validate_candidate=lambda checkpoint: True,
        )
        return scheduler, buffer, callbacks

    def test_cancellation_restores_evidence_and_allows_retry(self):
        for callback in ("train_candidate", "validate_candidate"):
            for error_type in (KeyboardInterrupt, SystemExit):
                with self.subTest(callback=callback, error=error_type):
                    scheduler, buffer, callbacks = self.make_update()
                    original = callbacks[callback]
                    error = error_type("cancelled")
                    callbacks[callback] = Mock(side_effect=error)
                    with self.assertRaises(error_type) as caught:
                        scheduler.try_update(**callbacks)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(len(buffer), 1)
                    self.assertEqual(scheduler.active, AdapterVersion(0, "initial"))
                    self.assertEqual(scheduler.pending_sessions, 1)
                    callbacks[callback] = original
                    self.assertTrue(scheduler.try_update(**callbacks))
                    self.assertEqual(len(buffer), 0)

    def test_restore_failure_preserves_original_error_context(self):
        scheduler, _, callbacks = self.make_update()
        cancellation = KeyboardInterrupt("cancelled")
        callbacks["train_candidate"] = Mock(side_effect=cancellation)
        callbacks["return_batch"] = Mock(side_effect=RuntimeError("restore failed"))
        with self.assertRaises(RuntimeError) as caught:
            scheduler.try_update(**callbacks)
        self.assertIs(caught.exception.__context__, cancellation)
        self.assertEqual(scheduler.active.version, 0)

    def test_whole_graph_quota_wait_is_atomic(self):
        buffer = CrossClientPreferenceBuffer()
        modalities = PaperMethodConfig().modalities
        for modality in modalities:
            for index in range(10):
                graph = consistent_pair_graph(
                    [0.9, 0.8, 0.7, 0.6, 0.5],
                    [PairJudgment(i, j, 0.9, 0.9) for i in range(5) for j in range(i + 1, 5)],
                    group_id=str(index),
                )
                self.assertEqual(len(graph.selected_edges), 10)
                buffer.add_graph(client_id=modality, modality=modality,
                                 policy_version=0, graph=graph)
        self.assertIsNone(buffer.take_balanced_edge_batch())
        self.assertEqual(len(buffer), 50)
        self.assertEqual(buffer.edge_count, 500)

    def test_torch_rejects_nonfinite_log_probabilities(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch dependency is not installed")
        from psyevo.lipo import torch_graph_lipo_loss
        for value in (float("nan"), float("inf"), -float("inf")):
            for invalid_reference in (False, True):
                good, bad = torch.tensor([-1., -2.]), torch.tensor([value, -2.])
                policy, reference = (good, bad) if invalid_reference else (bad, good)
                with self.subTest(value=value, reference=invalid_reference), self.assertRaises(ValueError):
                    torch_graph_lipo_loss(policy, reference, [PreferenceEdge(0, 1, 1.)], beta=0.05)


if __name__ == "__main__":
    unittest.main()
