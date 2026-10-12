import unittest

from psyevo import (
    AdapterVersion, CrossClientPreferenceBuffer, PaperMethodConfig,
    PopulationSessionScheduler, PairJudgment, consistent_pair_graph,
)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = PaperMethodConfig()
        self.scheduler = PopulationSessionScheduler(AdapterVersion(0, "initial"))
        self.buffer = CrossClientPreferenceBuffer()

    def fill(self, count=64):
        for modality in self.config.modalities:
            for i in range(count):
                graph = consistent_pair_graph(
                    pointwise_scores=[0.7, 0.4],
                    pair_judgments=[PairJudgment(0, 1, 0.9, 0.9)],
                    group_id=str(i),
                )
                self.buffer.add_graph(client_id=modality, modality=modality,
                                      policy_version=0, graph=graph)

    def complete(self, count=5):
        for i in range(count):
            self.scheduler.begin_session(str(i), "1")
            self.scheduler.complete_session(str(i), "1")

    def update(self, train=None, validate=None):
        return self.scheduler.try_update(
            reserve_batch=self.buffer.take_balanced_edge_batch,
            return_batch=self.buffer.return_batch,
            train_candidate=train or (lambda batch, old, config: "candidate"),
            validate_candidate=validate or (lambda path: True),
        )

    def test_paper_model_configuration(self):
        self.assertEqual((self.config.counselor, self.config.retrieval,
                          self.config.simulated_client, self.config.lora_rank),
                         ("Qwen3-32B", "BGE-M3", "DeepSeek-V3.1", 8))

    def test_four_sessions_do_not_consume_evidence(self):
        self.fill()
        self.complete(4)
        self.assertFalse(self.update())
        self.assertEqual(len(self.buffer), 320)

    def test_cross_client_count_and_version_pinning(self):
        self.fill()
        pinned = self.scheduler.begin_session("ongoing", "1")
        self.complete()
        self.assertFalse(self.scheduler.complete_session("0", "1"))
        self.assertEqual(self.scheduler.pending_sessions, 5)
        self.assertTrue(self.update())
        self.assertEqual(self.scheduler.active.version, 1)
        self.assertEqual(self.scheduler.session_adapter("ongoing", "1"), pinned)
        self.assertEqual(self.scheduler.begin_session("new", "1").version, 1)
        self.assertEqual(self.scheduler.pending_sessions, 0)

    def test_insufficient_evidence_keeps_gate_ready(self):
        self.fill(63)
        self.complete()
        self.assertFalse(self.update())
        self.assertEqual(len(self.buffer), 315)
        self.assertTrue(self.scheduler.eligible)

    def test_exception_restores_batch_and_active_version(self):
        self.fill()
        self.complete()
        def fail(*args):
            raise RuntimeError("training failed")
        with self.assertRaises(RuntimeError):
            self.update(train=fail)
        self.assertEqual(len(self.buffer), 320)
        self.assertEqual(self.scheduler.active.version, 0)
        self.assertTrue(self.scheduler.eligible)

    def test_rejection_restores_batch(self):
        self.fill()
        self.complete()
        self.assertFalse(self.update(validate=lambda path: False))
        self.assertEqual(len(self.buffer), 320)
        self.assertEqual(self.scheduler.active.version, 0)

    def test_no_in_place_checkpoint(self):
        self.fill()
        self.complete()
        with self.assertRaises(ValueError):
            self.update(train=lambda batch, old, config: old.checkpoint)
        self.assertEqual(len(self.buffer), 320)

    def test_exact_edge_quota_and_atomic_failure(self):
        self.fill()
        batch = self.buffer.take_balanced_edge_batch()
        self.config.check_batch(batch)
        self.assertEqual(batch.edge_count, 320)
        self.assertEqual(batch.modality_distribution, dict.fromkeys(self.config.modalities, 64))
        self.buffer.return_batch(batch)
        self.assertIsNone(self.buffer.take_balanced_edge_batch(min_groups=321))
        self.assertEqual(len(self.buffer), 320)


if __name__ == "__main__":
    unittest.main()
