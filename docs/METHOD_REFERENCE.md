# Method reference

This document maps the paper's mathematical objects to the supplementary code.

## 1. Learning state

For client `u`, PsyEvo separates a frozen shared skill table `Q`, a
client-private skill posterior `b_u`, and a population response adapter
`phi`. Candidate responses are compared only within one client and one state.
Only complete client-local SOCA graphs are pooled across clients.

## 2. HBSP

The skill value is

```text
Q_u(s, a) = Q(s, a) + b_u(s, a),
b_u(s, a) ~ Normal(m_u(s, a), lambda_u(s, a)^-1).
```

`HierarchicalBayesianSkillPolicy.action_probabilities` draws a posterior
residual when Thompson sampling is enabled, applies the paper's temperature
scaled softmax, and mixes it with uniform exploration. For the realized
trajectory return `G_t`, `update` uses

```text
d_t = G_t - Q(s_t, a_t)
lambda'_u = lambda_u + sigma_b^-2
m'_u = (lambda_u * m_u + sigma_b^-2 * d_t) / lambda'_u.
```

The shared table remains fixed, and updating one client does not modify any
other client's posterior.

## 3. SOCA candidate preferences

For each unordered pair `{i, j}`, `PairJudgment.forward_i_share` and
`reverse_i_share` both express support for the original candidate `i` under
the two presentation orders. A direction is retained only when both orders
agree strictly. Its confidence is

```text
c_ij = 2 * min(p_forward, p_reverse) - 1
```

for `i > j`, with the symmetric complement for `j > i`.
`consistent_pair_graph` sorts retained directions by confidence and adds an
edge only when doing so preserves acyclicity. `select_candidate` chooses the
maximum graph depth, then uses the pointwise score and candidate index only as
deterministic tie-breakers.

## 4. Realized-trajectory credit

`ordinal_credit` gives exact ties the same midrank and tier. It maps tiers to
ordinal levels and solves for a common shift before clipping to `[0, 1]`, so
the projected credits preserve the original session mean.
`discounted_returns` computes

```text
G_t = r_tilde_t + gamma * G_(t+1)
G_(T+1) = r_session.
```

Only these realized-trajectory returns update HBSP. Judgments about unshown
candidates are used only by the response learner.

## 5. Cross-client Population LiPO

`policy_reference_utilities` computes

```text
h_i = beta * (log pi_policy(y_i | s, a) - log pi_ref(y_i | s, a)).
```

For one client-local graph, `graph_lipo_objective` evaluates

```text
L_(u,g) = (1 / |E_(u,g)|) *
          sum_((i,j) in E_(u,g)) w_ij * softplus(-(h_i - h_j)).
```

For represented clients `U_B` and admitted groups `G_u`,
`population_graph_lipo_objective` evaluates

```text
L_pop = (1 / sum_(u in U_B) |G_u|) *
        sum_(u in U_B) sum_(g in G_u) L_(u,g).
```

Each admitted candidate group receives equal weight. Selected-edge
normalization prevents denser graphs from receiving more weight solely
because they contain more edges; client contributions follow their number
of admitted groups. `CrossClientPreferenceBuffer`
constructs deterministic round-robin batches with minimum client coverage,
required modality coverage, and an optional per-client cap. Construction is
atomic, and `return_batch` restores a complete batch after a failed update.
Each buffered group also retains the shared policy version that generated its
candidates.

The NumPy objective returns exact derivatives with respect to the utilities.
The optional `torch_graph_lipo_loss` and
`torch_population_graph_lipo_loss` retain autograd connections to policy
log-probabilities. The reference log-probabilities should be computed from the
frozen backbone with the shared adapter disabled.

## 6. Paper-to-code map

| Paper object | Public symbol |
|---|---|
| Skill posterior and exploration mixture | `HierarchicalBayesianSkillPolicy.action_probabilities` |
| Normal-Normal posterior update | `HierarchicalBayesianSkillPolicy.update` |
| Two-order agreement and edge confidence | `PairJudgment`, `consistent_pair_graph` |
| Acyclic weighted preference graph | `SOCAGraph.selected_edges` |
| Delivered-response selection | `select_candidate` |
| Tie-aware ordinal projection | `ordinal_credit` |
| Backward discounted credit | `discounted_returns` |
| Policy/reference utility | `policy_reference_utilities` |
| Per-graph weighted LiPO loss | `graph_lipo_objective`, `torch_graph_lipo_loss` |
| Balanced cross-client graph batch | `CrossClientPreferenceBuffer.take_batch` |
| Equal-group population objective | `population_graph_lipo_objective` |
| Autograd population loss | `torch_population_graph_lipo_loss` |
| Five-session population eligibility | `PopulationSessionScheduler.complete_session` |
| Exact modality-balanced edge budget | `CrossClientPreferenceBuffer.take_balanced_edge_batch` |
| Session-pinned adapter reference | `PopulationSessionScheduler.begin_session` |
| Paper model and rank-8 settings | `PaperMethodConfig` |

The runtime layer implements reference-level lifecycle control. It does not
bundle model inference or a checkpoint trainer. See `RUNTIME.md` for backend
requirements and failure semantics; no benchmark or experiment records are
needed to run the method checks.
