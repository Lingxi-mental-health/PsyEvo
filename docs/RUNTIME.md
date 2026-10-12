# Model integration and session lifecycle

Use one `PopulationSessionScheduler` and one `CrossClientPreferenceBuffer` per
shared learning scope. Do not instantiate a scheduler for each client. The
default configuration matches the paper's five-session eligibility interval,
not an unconditional training operation every five sessions.

```python
from psyevo import (
    AdapterVersion, CrossClientPreferenceBuffer, PaperMethodConfig,
    PopulationSessionScheduler,
)

config = PaperMethodConfig()
buffer = CrossClientPreferenceBuffer()
scheduler = PopulationSessionScheduler(AdapterVersion(0, "initial-adapter"), config)

# The application's backend supplies model generation and training functions.
pinned = scheduler.begin_session("client-a", "session-1")
# Generate all responses with pinned.checkpoint; add complete local SOCA graphs.
# After the realized session: update this client's HBSP posterior and memory.
scheduler.complete_session("client-a", "session-1")

# Reuse this scheduler for subsequent sessions from every client in the scope.
# This call does nothing before five completed sessions, or without enough data.
def update_at_boundary(train_candidate, validate_candidate):
    return scheduler.try_update(
        reserve_batch=lambda: buffer.take_balanced_edge_batch(
            pairs=config.preference_pairs,
            min_groups=config.min_groups,
            modalities=config.modalities,
        ),
        return_batch=buffer.return_batch,
        train_candidate=train_candidate,
        validate_candidate=validate_candidate,
    )
```

## Backend contract

`train_candidate(batch, active, config)` must train a separate rank-8 adapter
initialized from `active.checkpoint`, leaving the active checkpoint and all
existing sessions untouched. It returns the new checkpoint identifier. Use
`torch_population_graph_lipo_loss` with differentiable Qwen3-32B policy log
probabilities and detached reference log probabilities from the same frozen
backbone with the adapter disabled. Match response token masks for both.
The backend supplies prompts, response tokens and local evidence associated
with each `(client_id, group_id)`; the numerical method package does not store
or generate these texts.

Use learning rate 1e-5, beta 0.05, batch size 2, gradient accumulation 4,
maximum sequence length 1024, at most 3 optimizer steps and 1 epoch per round.
These values are passed as configuration; this package does not load Qwen or
instantiate a PEFT trainer. Validate trainable parameters and LoRA rank in the
backend rather than treating the metadata as proof of a correctly loaded model.

`validate_candidate(checkpoint)` checks loading and numerical validity before
the coordinator accepts the identifier. Acceptance changes the reference for
new sessions only. Old sessions retain their captured identifier. Failed
training or validation restores the complete batch and keeps the old active
reference. Candidate cleanup, durable snapshots, optimizer-state recovery,
serving publication and restart recovery remain backend responsibilities.
Returning evidence is not physical model rollback. Callbacks must not mutate
the active checkpoint, re-enter the scheduler or edit the reserved evidence.
Catchable cancellation during training or validation (including KeyboardInterrupt
and SystemExit) also restores the reserved batch and is then propagated.
If restoration itself fails, its exception propagates with the original failure
as exception context; the backend must resolve that failed restoration before retrying.

BGE-M3 is the retrieval embedding model. Supply its retrieved context to the
counselor's prompt construction. DeepSeek-V3.1 is the simulated-client model;
it is an external interaction backend, not an external evaluator or a source
of cross-client preference edges. This release provides no benchmark-specific
client profiles or simulation/evaluation prompts.

## Batching and concurrency

The reference sampler retains whole SOCA graphs and finds exactly 64 retained
edges per modality (five modalities, 320 total). It never cuts a graph to meet
a quota. If no exact full-graph selection exists, it waits without consuming
evidence. The sampler is deterministic; it does not reproduce a particular
experimental interleaving or pair-reservoir sampling order. For example, any
number of ten-edge graphs cannot meet a 64-edge quota: more evidence alone does
not guarantee eligibility. This waiting case preserves all buffered graphs.
Candidate groups
have equal loss weight, not clients. SOCA edges remain within one client/state.

Scheduler calls are serialized. Generation may proceed using captured immutable
checkpoint identifiers, but shared-buffer writes must be serialized by the
application with update reservation/return. The buffer itself is not thread-safe.
No durable resume is implemented; create a fresh learning scope for a new run.
