# PsyEvo method implementation

This anonymous supplementary archive contains the method-level implementation
for **PsyEvo: A Personalized Counseling Agent That Self-Evolves at Test Time**.
It implements the learning pipeline described in the paper:

- **HBSP** maintains a client-private Normal posterior over skill residuals;
- **SOCA** constructs consistency-checked, acyclic candidate preferences and
  tie-aware credit for the realized trajectory;
- **Population LiPO** pools complete client-local SOCA graphs and optimizes one
  shared response adapter with equal weight per candidate group.

The implementation is backend-agnostic. Candidate generation, pairwise
judgment, token log-probability computation, and adapter optimization are
explicit integration boundaries. No task-specific data or evaluation code is
included.

This is a standalone reference implementation of the method, not a verbatim
export of the experimental training system. The included example evaluates
the cross-client objective on synthetic inputs. It does not train or publish
an adapter checkpoint. The package implements population session scheduling,
batch eligibility checks and session-level adapter-reference pinning. Model
inference, retrieval, physical checkpoint creation/loading and memory
summarization require a backend. Training must create an isolated candidate:
failure leaves the old active reference unchanged and restores evidence; this
does not undo arbitrary in-place backend parameter changes.

## Paper model configuration

`PaperMethodConfig` records the implementation choices described in the paper:

| Role | Configuration |
|---|---|
| Counseling backbone | Qwen3-32B, frozen |
| Retrieval embeddings | BGE-M3 |
| Simulated client | DeepSeek-V3.1 |
| Trainable response module | Shared rank-8 LoRA adapter |
| Objective | SOCA-weighted, edge-normalized, equal-group Population LiPO |
| Reference policy | Frozen backbone with adapter disabled |
| Eligibility interval | 5 completed sessions across clients sharing the adapter |
| Evidence gate | 320 retained edges, at least 32 groups, 20% edges per modality |

The model names are configuration metadata, not bundled weights or functioning
API clients. `train_candidate` receives this configuration; the backend must
enforce the rank, frozen backbone and optimizer settings. No dataset loader,
benchmark evaluator, external judge, experimental log or credential is included.
Dataset statistics, baseline comparisons and cost reports belong in the paper,
not in this method package. See `docs/RUNTIME.md` for executable integration.

## Contents

```text
PsyEvo-method/
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.cfg
├── requirements.txt
├── docs/
│   ├── METHOD_REFERENCE.md
│   └── RUNTIME.md
├── examples/
│   └── minimal_pipeline.py
├── src/psyevo/
│   ├── __init__.py
│   ├── hbsp.py
│   ├── soca.py
│   ├── population.py
│   ├── lipo.py
│   └── runtime.py
└── tests/
    ├── test_method.py
    ├── test_runtime.py
    └── test_robustness.py
```

## Requirements

- Python 3.9 or newer
- NumPy 1.24 or newer
- PyTorch 2.0 or newer only for the optional autograd loss wrappers

## Install

From the archive root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

For the optional PyTorch wrappers:

```bash
python3 -m pip install -e '.[torch]'
```

## Run the method example

```bash
python3 examples/minimal_pipeline.py
```

The example creates two client-local SOCA graphs, forms a cross-client batch,
evaluates the equal-group Population LiPO objective, and applies one HBSP
posterior update from realized trajectory credit.

## Run the checks

```bash
python3 -m unittest discover -s tests -v
```

The checks cover private HBSP posteriors, SOCA abstention and acyclicity,
mean-preserving trajectory credit, discounted returns, analytic LiPO
gradients, client/modality coverage, atomic batching, failed-batch return, and
equal-group population weighting.

## Method boundaries

```text
same client + same state -> candidate judgments -> local SOCA graph
local graphs from clients -> balanced population batch -> Population LiPO
realized client trajectory -> SOCA credit -> returns -> private HBSP update
```

SOCA never creates an edge between candidates from different clients. The
population buffer pools complete local graphs while retaining client and
modality identities and the policy version that generated each group. Each
graph loss is normalized by its retained edge count; group losses are averaged
over all admitted groups in the batch. A model integration should pin one
accepted adapter version for an entire session and publish a new version only
after a successful between-session update.

See `docs/METHOD_REFERENCE.md` for the paper-to-code mapping and equations.
