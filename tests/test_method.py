import unittest

import numpy as np

from psyevo import (
    CrossClientPreferenceBuffer,
    HBSPConfig,
    HierarchicalBayesianSkillPolicy,
    PairJudgment,
    PreferenceEdge,
    consistent_pair_graph,
    discounted_returns,
    graph_lipo_objective,
    ordinal_credit,
    population_graph_lipo_objective,
    select_candidate,
)


class HBSPTests(unittest.TestCase):
    def test_posterior_contracts_and_clients_are_isolated(self):
        policy = HierarchicalBayesianSkillPolicy(
            np.array([[0.2, 0.4], [0.1, 0.3]]),
            HBSPConfig(
                prior_variance=0.25,
                observation_variance=1.0,
                thompson_sampling=False,
                epsilon=0.0,
            ),
        )
        before_sd = policy.posterior_sd("client-a", 0)[1]
        update = policy.update("client-a", 0, 1, trajectory_return=0.8)
        after_sd = policy.posterior_sd("client-a", 0)[1]

        self.assertAlmostEqual(before_sd, 0.5)
        self.assertLess(after_sd, before_sd)
        self.assertAlmostEqual(update.residual, 0.4)
        self.assertAlmostEqual(update.posterior_precision, 5.0)
        self.assertEqual(policy.posterior_mean("client-b", 0)[1], 0.0)

    def test_invalid_actions_receive_zero_probability(self):
        policy = HierarchicalBayesianSkillPolicy(
            np.zeros((1, 3)),
            HBSPConfig(thompson_sampling=False, epsilon=0.2),
        )
        probabilities, _, _, _ = policy.action_probabilities(
            "client", 0, valid_actions=[0, 2]
        )
        np.testing.assert_allclose(probabilities, [0.5, 0.0, 0.5])


class SOCATests(unittest.TestCase):
    def test_order_conflict_abstains_and_graph_is_acyclic(self):
        graph = consistent_pair_graph(
            [0.4, 0.9, 0.6],
            [
                PairJudgment(0, 1, 0.0, 0.0),
                PairJudgment(0, 2, 0.25, 0.25),
                PairJudgment(1, 2, 0.75, 0.25),
            ],
            group_id="fixture",
        )
        self.assertEqual(graph.stable_edge_count, 2)
        self.assertEqual(
            graph.selected_edges,
            (
                PreferenceEdge(1, 0, 1.0),
                PreferenceEdge(2, 0, 0.5),
            ),
        )
        self.assertEqual(select_candidate(graph), 1)

    def test_pointwise_scores_do_not_change_soca_edges(self):
        judgments = [
            PairJudgment(0, 1, 0.9, 0.9),
            PairJudgment(0, 2, 0.8, 0.8),
            PairJudgment(1, 2, 0.7, 0.7),
        ]
        first = consistent_pair_graph([0.1, 0.2, 0.3], judgments, group_id="same-group")
        second = consistent_pair_graph(
            [0.9, 0.2, 0.1], judgments, group_id="same-group"
        )
        self.assertEqual(first.selected_edges, second.selected_edges)

    def test_ordinal_projection_preserves_mean_and_ties(self):
        result = ordinal_credit([0.0, 0.0, 0.9, 1.0], buckets=3, spread=0.8)
        self.assertAlmostEqual(
            sum(result.credits) / len(result.credits), result.mean, places=12
        )
        self.assertEqual(result.credits[0], result.credits[1])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in result.credits))

    def test_discounted_returns_use_terminal_bootstrap(self):
        returns = discounted_returns([0.2, 0.4], terminal_reward=0.5, gamma=0.9)
        np.testing.assert_allclose(returns, [0.2 + 0.9 * (0.4 + 0.9 * 0.5), 0.85])

    def test_hbsp_accepts_cumulative_return(self):
        policy = HierarchicalBayesianSkillPolicy(np.array([[0.2, 0.4]]))
        update = policy.update("client", 0, 0, trajectory_return=1.4)
        self.assertAlmostEqual(update.residual, 1.2)


class LiPOTests(unittest.TestCase):
    def test_analytic_utility_gradient_matches_finite_difference(self):
        policy = np.array([-2.0, -1.1, -1.6])
        reference = np.array([-2.2, -1.4, -1.7])
        edges = (PreferenceEdge(1, 0, 0.8), PreferenceEdge(2, 0, 0.4))
        beta = 0.2
        loss, utility_gradient = graph_lipo_objective(
            policy_logp=policy,
            reference_logp=reference,
            edges=edges,
            beta=beta,
        )

        epsilon = 1e-6
        numerical = np.zeros(3)
        # Perturb policy log-probability by epsilon / beta so utilities move by
        # exactly epsilon; this checks d(loss)/d(utility).
        for index in range(3):
            plus = policy.copy()
            minus = policy.copy()
            plus[index] += epsilon / beta
            minus[index] -= epsilon / beta
            plus_loss, _ = graph_lipo_objective(
                policy_logp=plus,
                reference_logp=reference,
                edges=edges,
                beta=beta,
            )
            minus_loss, _ = graph_lipo_objective(
                policy_logp=minus,
                reference_logp=reference,
                edges=edges,
                beta=beta,
            )
            numerical[index] = (plus_loss - minus_loss) / (2.0 * epsilon)

        self.assertGreater(loss, 0.0)
        np.testing.assert_allclose(utility_gradient, numerical, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(float(utility_gradient.sum()), 0.0, places=12)


class CrossClientLiPOTests(unittest.TestCase):
    @staticmethod
    def graph(group_id, winner=0):
        return consistent_pair_graph(
            [0.7, 0.3],
            [
                PairJudgment(
                    0,
                    1,
                    0.9 if winner == 0 else 0.1,
                    0.9 if winner == 0 else 0.1,
                )
            ],
            group_id=group_id,
        )

    def test_batch_is_cross_client_balanced_and_keeps_edges_local(self):
        buffer = CrossClientPreferenceBuffer()
        buffer.add_graph(
            client_id="a", modality="cbt", policy_version=0, graph=self.graph("a-1")
        )
        buffer.add_graph(
            client_id="a", modality="cbt", policy_version=0, graph=self.graph("a-2")
        )
        buffer.add_graph(
            client_id="b", modality="pdt", policy_version=1, graph=self.graph("b-1", 1)
        )
        buffer.add_graph(
            client_id="c", modality="bt", policy_version=1, graph=self.graph("c-1")
        )

        batch = buffer.take_batch(
            min_groups=3,
            max_groups=4,
            min_clients=3,
            required_modalities=["cbt", "pdt", "bt"],
            max_groups_per_client=2,
        )
        self.assertEqual(set(batch.client_distribution), {"a", "b", "c"})
        self.assertEqual(set(batch.modality_distribution), {"cbt", "pdt", "bt"})
        self.assertEqual(batch.edge_count, 4)
        self.assertEqual({group.policy_version for group in batch.groups}, {0, 1})
        self.assertEqual(len(buffer), 0)
        self.assertTrue(
            all(
                group.graph.group_id.startswith(group.client_id)
                for group in batch.groups
            )
        )
        buffer.return_batch(batch)
        self.assertEqual(len(buffer), 4)
        with self.assertRaisesRegex(ValueError, "duplicate client/group key"):
            buffer.return_batch(batch)
        self.assertEqual(len(buffer), 4)

    def test_failed_batch_is_atomic(self):
        buffer = CrossClientPreferenceBuffer()
        buffer.add_graph(
            client_id="a", modality="cbt", policy_version=0, graph=self.graph("a-1")
        )
        with self.assertRaisesRegex(ValueError, "required modality"):
            buffer.take_batch(
                min_groups=1,
                max_groups=2,
                min_clients=1,
                required_modalities=["pdt"],
            )
        self.assertEqual(len(buffer), 1)
        buffer.add_graph(
            client_id="b", modality="pdt", policy_version=0, graph=self.graph("b-1")
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed max_groups"):
            buffer.take_batch(
                min_groups=1,
                max_groups=1,
                min_clients=1,
                required_modalities=["cbt", "pdt"],
            )
        self.assertEqual(len(buffer), 2)

    def test_modality_reservation_does_not_repeat_an_already_covered_client(self):
        buffer = CrossClientPreferenceBuffer()
        for client, group_id, modality in (
            ("a", "a-1", "cbt"), ("a", "a-2", "cbt"), ("b", "b-1", "pdt")
        ):
            buffer.add_graph(client_id=client, modality=modality,
                             policy_version=0, graph=self.graph(group_id))
        batch = buffer.take_batch(max_groups=2, min_clients=2,
                                  required_modalities=["cbt"])
        self.assertEqual(batch.client_distribution, {"a": 1, "b": 1})
        self.assertEqual(len(buffer), 1)

    def test_population_loss_weights_groups_equally_with_unequal_client_counts(self):
        buffer = CrossClientPreferenceBuffer()
        for group_id in ("a-1", "a-2"):
            buffer.add_graph(
                client_id="a",
                modality="cbt",
                policy_version=0,
                graph=self.graph(group_id),
            )
        buffer.add_graph(
            client_id="b", modality="pdt", policy_version=0, graph=self.graph("b-1", 1)
        )
        batch = buffer.take_batch(min_groups=3, max_groups=3, min_clients=2)
        policy = {
            ("a", "a-1"): [-1.0, -2.0],
            ("a", "a-2"): [-1.0, -2.0],
            ("b", "b-1"): [-1.0, -2.0],
        }
        reference = {key: [-1.5, -1.5] for key in policy}
        result = population_graph_lipo_objective(
            batch=batch,
            policy_logp=policy,
            reference_logp=reference,
            beta=0.2,
        )
        a_loss = result.group_losses[("a", "a-1")]
        b_loss = result.group_losses[("b", "b-1")]
        self.assertAlmostEqual(result.loss, (2.0 * a_loss + b_loss) / 3.0)
        _, raw_a_gradient = graph_lipo_objective(
            policy_logp=policy[("a", "a-1")],
            reference_logp=reference[("a", "a-1")],
            edges=next(
                group.graph.selected_edges
                for group in batch.groups
                if group.key == ("a", "a-1")
            ),
            beta=0.2,
        )
        _, raw_b_gradient = graph_lipo_objective(
            policy_logp=policy[("b", "b-1")],
            reference_logp=reference[("b", "b-1")],
            edges=next(
                group.graph.selected_edges
                for group in batch.groups
                if group.key == ("b", "b-1")
            ),
            beta=0.2,
        )
        np.testing.assert_allclose(
            result.utility_gradients[("a", "a-1")], raw_a_gradient / 3.0
        )
        np.testing.assert_allclose(
            result.utility_gradients[("b", "b-1")], raw_b_gradient / 3.0
        )

    def test_population_autograd_matches_numpy_and_detaches_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch dependency is not installed")
        from psyevo import torch_population_graph_lipo_loss
        buffer = CrossClientPreferenceBuffer()
        for client, group_id in (("a", "a-1"), ("a", "a-2"), ("b", "b-1")):
            buffer.add_graph(client_id=client, modality="fixture", policy_version=0,
                             graph=self.graph(group_id, winner=int(client == "b")))
        batch = buffer.take_batch(max_groups=3, min_clients=2)
        policy = {g.key: torch.tensor([-1., -2.], dtype=torch.float64,
                                     requires_grad=True) for g in batch.groups}
        reference = {g.key: torch.tensor([-1.5, -1.5], dtype=torch.float64,
                                        requires_grad=True) for g in batch.groups}
        expected = population_graph_lipo_objective(
            batch=batch, policy_logp={k: v.detach().numpy() for k,v in policy.items()},
            reference_logp={k: v.detach().numpy() for k,v in reference.items()}, beta=0.2)
        loss = torch_population_graph_lipo_loss(
            batch=batch, policy_logp=policy, reference_logp=reference, beta=0.2)
        self.assertAlmostEqual(loss.item(), expected.loss)
        loss.backward()
        for key in policy:
            np.testing.assert_allclose(policy[key].grad.numpy(),
                                       0.2 * expected.utility_gradients[key])
            self.assertIsNone(reference[key].grad)


if __name__ == "__main__":
    unittest.main()
