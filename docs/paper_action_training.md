# Paper action fitting

The training entry is `scripts/paper_training/train_action.py`. It implements the continuation fit that produced the published mixed-data actor. It learns the student's representation and action head, with 943,685 trainable parameters. It does not tune Astra, train a new waiting controller or change the stopping rule.

## Data and initialization

A separately supplied `paper_action_data_v1` bundle is required. This repository does not download or include that bundle. The author-side export command is:

```bash
python scripts/paper_training/export_data.py \
  --research-root /path/to/original-research-checkout \
  --output /path/to/action-data
```

The original research checkout must contain the completed collection and its source/data provenance. Export first runs the original readiness checks, then copies exactly four tensors per shard: `rgb_history`, `proprio_history`, `goal_id` and `teacher_action`. It does not filter, convert, resample or relabel examples. The output manifest links every exported shard to its original hash and split. No service traces, raw images or credentials are exported.

| Domain | Training samples | Validation samples | Training / validation reset seeds |
| --- | ---: | ---: | --- |
| Earlier teacher-labelled data | 11,634 | 1,528 | 14 / 2 |
| New student-visited, teacher-labelled data | 16,485 | 6,339 | 6 / 2 |

These 35,986 observation histories are correlated samples, not independent episodes. The bundle contains 24 shards and the required prior fitted checkpoint, SHA256 `b000c4bc027ea3eb1a442c1b925f66002bddc98c1375116575693cb5bf249187`. Both fitted arms start from that prior checkpoint. The published reference student supplies the frozen action anchor. Starting from the final mixed checkpoint would change the experiment.

The required prior checkpoint is kept in the external data bundle, not added as another default runtime weight. Dataset distribution and raw-data regeneration remain separate from this code release.

## Fixed recipe

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/paper_training/train_action.py \
  --data-dir /path/to/action-data \
  --output outputs/paper_action_fit
```

The command acquires the shared GPU lock and runs exactly 50 epochs, 182 updates per epoch. Seed is 146001, with TF32 disabled. Both arms share the first 32 old samples in each batch. Old-only adds 32 old samples, while mixed adds 32 new samples. All labels are teacher actions. There is no failure weighting.

The loss is Smooth-L1 to the teacher with beta 0.1 plus 0.05 times mean-squared distance to the frozen reference action. AdamW uses learning rates 1e-4 for the action head and 1e-5 for the representation, weight decay 1e-4 and gradient clipping at 1. Evidence and stop head parameters are frozen. Their outputs can still change through the representation, so the evaluator discards candidate evidence/stop outputs and retains the reference feedback path.

Validation weights seeds equally within each domain and the two domains equally. Each arm independently selects the lowest held-out teacher loss. Rollout success does not select the epoch. The original mixed fit selected epoch 47. The command saves both fitted arms, their selection records and data/code hashes. It does not launch evaluation, call a remote model or deploy a checkpoint.

## Verified scope

```bash
CUDA_VISIBLE_DEVICES='' python scripts/paper_training/check.py \
  --research-root /path/to/original-research-checkout \
  --data-dir /path/to/action-data
```

The release check confirms exact AST equality for the entire epoch loop, optimizer setup, batching and validation against the original trainer. All 24 exported shards pass the loader. CPU validation on all 7,867 held-out histories reproduces the recorded baseline losses with maximum absolute difference zero on the checked host. One CPU optimizer update in each arm changes the action head and leaves evidence/stop head parameters unchanged. Both arms have 943,685 trainable parameters.

`paper_training_verification.json` records source hashes, data identity and check results. A complete 50-epoch fit was not repeated for this release. These checks do not establish cross-hardware bitwise reproducibility or add a task-success result. Runtime weights and evaluation definitions remain unchanged.
