# Deep Learning Module Integration Plan

## Source Files

| Prototype | Location |
|-----------|----------|
| `start_v3.py` | `ps_prototypes_v2/start_v3.py` (PyTables-based baseline) |
| `start_v4_sqlite.py` | `ps_prototypes_v2/start_v4_sqlite.py` (SQLite + candidate pool enrichment) |
| `sqlite_dataset.py` | `ps_prototypes_v2/sqlite_dataset.py` |
| `utils.py` | `ps_prototypes_v2/utils.py` (SwAV loss, adaptive threshold, scoring, LabeledRateTracker) |
| `configs.py` | `ps_prototypes_v2/configs.py` |

| Production | Location |
|------------|----------|
| `losses.py` | `patchsorter/dl/losses.py` |
| `training.py` | `patchsorter/dl/training.py` |
| `model.py` | `patchsorter/dl/model.py` |
| `augmentations.py` | `patchsorter/dl/augmentations.py` |

---

## 1. Differences: `start_v4_sqlite.py` vs `start_v3.py`

### 1.1 Database backend
- **v3**: PyTables (`.pytable`) via `tables` library with custom `Dataset` class
- **v4**: SQLite with `SQLiteDataset`, `GTCandidatePool`, `CandidatePoolIterableDataset`, `ScoreWriter`

### 1.2 Candidate pool enrichment (new in v4)
- `GTCandidatePool` (line 172, `sqlite_dataset.py`) — in-memory candidate pool with decayed rarity scoring
- `CandidatePoolIterableDataset` (line 321, `sqlite_dataset.py`) — `IterableDataset` that yields enriched batches from the candidate pool
- Priority column (format `XX.YY`): XX = iteration count (times seen), YY = decayed rarity score (replaces `score_timestamp`)
- Partial index on `priority` where `priority > 0` (line 102-108, `sqlite_dataset.py`)
- `ScoreWriter` for writing scores back to the database

### 1.3 Loss function changes
| Aspect | v3 | v4 |
|--------|----|----|
| Embedding loss | `simclr_loss()` or `swav_loss()` (configurable via `LOSS_TYPE`) | `swav_loss()` always (config removed, SwAV is now canonical) |
| MMD loss | `max_mean_discrepancy()` with `MAX_MEAN_LOSS` weight | Removed — `max_mean_discrepancy()` and its weight constant are dropped from the training loop; the function may remain in `losses.py` for now but is no longer called |
| New loss | — | `rank_uniform_loss()` with `RANK_UNIFORM_LOSS` weight |
| Pseudo-label loss | `prediction_loss_pseudo()` (majority vote + fixed threshold) | `prediction_loss_pseudo_sce_adaptive()` (adaptive per-class threshold + SCE loss) |

### 1.4 Label tracker update scope fix
- **v3** line 363: `label_tracker.update(labels, ...)` — passes ALL labels including view repetitions
- **v4** line 394: `label_tracker.update(labels[0:nbase_ids], ...)` — passes only base batch labels (first view per patch)
- Pseudo labels filtered through `high_conf` mask before passing to tracker

### 1.5 View reconstruction bug fix (line 295)
- **v3** line 274: `torch.stack(views_gpu, dim=1).flatten(0, 1)` — wrong axis, produces `[B, V, ...]`
- **v4** line 295: `torch.stack(views_gpu, dim=0).flatten(0, 1)` — correct, produces `[V, B, ...] → [V*B, ...]`

This is the critical fix for labels from different views. The flattened layout must be `[V*B, ...]` (views-major) for losses like `simclr_loss` and `semantic_head_loss` that operate on the flat tensor with `labels.repeat(V)`.

### 1.6 Score computation (new in v4)
- `compute_weighting_scores()` (line 1470, `utils.py`) — combines spatial rarity (kNN distance in embedding space) with class rarity weights
- `ScoreWriter.enqueue()` writes `XX(niter_total).YYY(score)` format to `priority` column
- Only first-view base batch items scored (`nbase_ids`), not enriched candidates
- **Bug in prototype** (`start_v4_sqlite.py:400`): `compute_weighting_scores(ids, labels, proj_emb, ..., len(ids))` passes `len(ids)` as the count, but only the first `nbase_ids` elements of `ids`/`proj_emb` are from the sequential batch. The correct call is `compute_weighting_scores(ids, labels, proj_emb, ..., nbase_ids)`. This must be fixed in the production implementation.

### 1.7 Adaptive threshold (new in v4)
- `AdaptiveThreshold` class (line 1396, `utils.py`) — FlexMatch-style per-class adaptive thresholding
- Tracks `sigma_c` (EMA of per-batch crossing counts) for each class
- `high_conf_mask()` applies per-class thresholds to majority-vote confidence

---

## 2. Database Changes

### 2.1 Partial index on priority column

**Purpose**: Rapid selection of rare/labeled patches for training enrichment.

**Proposed index**:
```sql
CREATE INDEX idx_patches_priority_positive
ON patches (priority)
WHERE priority > 0;
```

**Notes**:
- No partial index needed on ground truth label column
- The prototype's partial index on `score_timestamp` (line 102-108, `sqlite_dataset.py`) shows the pattern — this will be migrated to `priority`

### 2.2 Priority column on patch table

**Format**: `XX.YY`
- `XX` — integer: number of times the patch has been sampled for training
- `YY` — float in `[0, 1]`: patch weighting (rarity score)

**Implementation**: Add `priority` column to the dynamic patch model in `patchsorter/db/head_client/models.py:153-172`.

**Action**: Add to the `patch_model()` function's column dict:
```python
"priority": Column(Float, nullable=True),
```

**Notes**:
- `priority` replaces `score_timestamp` from the prototype
- No `ALTER TABLE` migration needed — the column is defined in the ORM model
- Update interval: per-batch (like prototype)

---

## 3. Dataloader Design

### 3.1 `dataloader_sequential`

**Source**: `IterableShardDataset` (new class, based on `ShardDataset` pattern in `training.py`)

**Behavior**:
- Iterates sequentially through worker-assigned shards
- No enrichment, no candidate pool
- Used for prediction saving (only sequential part of batch is saved)
- Iterated until exhausted at the start of each cycle

**Implementation approach**:
- Extend or replace `ShardDataset` in `training.py` to be an `IterableDataset`
- Each worker gets a deterministic shard subset via `compute_shard_assignments()`
- Yields batches of decoded patches from assigned shards

### 3.2 `dataloader_enriched`

**Source**: `EnrichedInfiniteIterableDataset` (new class, based on `CandidatePoolIterableDataset` in `sqlite_dataset.py`)

**Behavior**:
- Uses in-memory `CandidatePool` with candidate scores
- Infinite iteration with periodic pool refresh
- Draws batches weighted by candidate score (rarity)
- Decays in-memory scores after each draw

**Implementation approach**:
- Create `EnrichedInfiniteIterableDataset` as a new `IterableDataset`
- Each worker owns its own `CandidatePool` (in-memory) + PostgreSQL worker client connection
- Pool refreshes every N batches from PostgreSQL via manual UNION across shards (not Citus) with `ORDER BY priority` + `LIMIT`
- `batch_size=None` on wrapping `DataLoader` — dataset yields pre-collated batches

**Backend**: Production uses PostgreSQL via the worker client (not SQLite). The prototype's `sqlite_dataset.py` pattern informs the design but the implementation queries PostgreSQL directly.

**Sharding note**: The enriched dataloader must query across Citus shards. Since Citus does not support `ORDER BY` + `LIMIT` across shards efficiently, the UNION must be performed manually on PostgreSQL (not through Citus). Each shard is queried separately and results merged client-side.

### 3.3 Training loop structure

```
# Outside cycle loop:
enriched_loader = DataLoader(EnrichedInfiniteIterableDataset(...), batch_size=None, num_workers=N)
enriched_iter = iter(enriched_loader)

# Inside cycle loop:
for finite_batch in dataloader_sequential:  # stops when exhausted
    infinite_batch = next(enriched_iter)
    batch = concat_batches(finite_batch, infinite_batch)
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Enriched batch size**: Controlled by `GT_ENRICHMENT = 0.10` — the enriched dataloader yields approximately 10% of the sequential batch size.

**Key design decisions**:
- Enriched dataloader instantiated outside sequential loop (as specified) to maintain pool state
- Each training batch = concatenation of sequential + enriched
- Only sequential part saved to predictions (per spec)
- Only base (sequential) batch labels update the label weight tracker — `labels[0:nbase_ids]` — matching `start_v4_sqlite.py:395`
- Both parts contribute to loss computation and backprop

---

## 4. Loss Function Updates

### 4.1 SwAV loss — rename and replace

**Current state**: `losses.py` line 14 has `simclr_loss()` which is NT-Xent (SimCLR). In the prototype, `LOSS_TYPE` was removed and the training loop was updated to always call `swav_loss()`, but `simclr_loss` was never renamed in the prototype's `utils.py`. The function at `utils.py:1191` is the true SwAV implementation (Sinkhorn prototype assignments) and is a different algorithm from NT-Xent.

**Action**: In production `losses.py`, rename `simclr_loss()` to `swav_loss()` and **replace its implementation** with the SwAV computation from `ps_prototypes_v2/utils.py:1191`. The NT-Xent implementation is no longer used.

**Scope**: SwAV is applied only to **embedding coordinates** (`proj_emb`). The production `training.py` also computes `simclr_coord_loss = simclr_loss(proj_coords, ...)` (NT-Xent on 2D projection coordinates) — this term is **removed entirely**. The prototype already comments this out (`start_v4_sqlite.py:357`: `#simclr_emb_loss_coord = swav_loss(proj_coords)  -- this seems crazy`). The corresponding `SIMCLR_EMB_LOSS * simclr_coord_loss` term is dropped from `total_loss`.

**Considerations**:
- The new `swav_loss()` accepts a `prototypes` parameter (learnable prototypes from `JointHead`) — add `prototypes` attribute to `JointHead` model
- Add `prototypes` as a `nn.Parameter` in `JointHead.__init__()` with initialization via `_init_weights()`: small normal init followed by `F.normalize(..., dim=1)` (unit-sphere constraint)
- Prototype normalization must be re-applied after each optimizer step (normalize `model.head.prototypes.data` in the training loop)
- SwAV parameters from `configs.py`: `SWAV_PROTOTYPES=300`, `SWAV_KMEANS_ITERS=10`, `SWAV_SINKHORN_ITERS=3`, `SWAV_EPS=0.05`, `temp=0.1`
- **Implementation**: Copy `swav_loss()` from `ps_prototypes_v2/utils.py:1191` as-is. Keep `SWAV_KMEANS_ITERS` as a variable (no need to re-implement k-means logic — the prototype's k-means initialization is sufficient).

### 4.2 `prediction_loss_pseudo_sce_adaptive` — integrate

**Current state**: `losses.py` has `prediction_loss_pseudo()` (majority vote + fixed threshold + standard CE).

**Action**: Add `prediction_loss_pseudo_sce_adaptive()` from `ps_prototypes_v2/utils.py:1331` to `losses.py`.

**Key differences from existing**:
- Uses `AdaptiveThreshold` for per-class confidence thresholds instead of fixed `pseudo_thresh`
- Uses SCE loss (symmetric cross-entropy) instead of standard CE
- Requires `num_classes` parameter (uses `N_CLASS` from configs)

### 4.3 `rank_uniform_loss` — add

**Action**: Add `rank_uniform_loss()` from prototype to `losses.py`.

**Current state**: Not present in `losses.py`. The prototype uses it with weight `RANK_UNIFORM_LOSS = 10_000.0`.

### 4.4 Loss weight parameters

All production constants in `training.py` must be updated to exactly match `ps_prototypes_v2/configs.py`. These values have been carefully tuned in the prototype and should be preserved as-is. The table below shows the full migration, including renames, value changes, and removals.

| Current production | Current value | Target constant | Target value | Action |
|--------------------|---------------|-----------------|--------------|--------|
| `SIMCLR_EMB_LOSS` | 100.0 | `SWAV_EMB_LOSS` | 100 | rename |
| `MAX_MEAN_LOSS` | 1000.0 | — | — | remove (MMD dropped) |
| `COORD_CONTRASTIVE_LOSS` | **1.0 (active)** | `COORD_CONTRASTIVE_LOSS` | **0** | value change — set to 0 to match prototype |
| `SEMANTIC_LAMBDA` | 1.0 (shared) | `SEMANTIC_COORD_LAMBDA` | 1.0 | split into two constants |
| `SEMANTIC_LAMBDA` | 1.0 (shared) | `SEMANTIC_EMB_LAMBDA` | **10.0** | split — emb weight is 10× higher than prod |
| `PRED_LAMBDA` | **100.0** | `PRED_SUP_LAMBDA` | **10_000.0** | rename + 100× increase |
| `PSEUDO_PRED_LAMBDA` | **0.4** | `PSEUDO_PRED_LAMBDA` | **0.0001** | value change (intermediate factor) |
| (new) | — | `PRED_PSEUDO_LAMBDA` | `PRED_SUP_LAMBDA × PSEUDO_PRED_LAMBDA = 1.0` | derived constant for pseudo loss term |
| `COORD_CONSISTENCY_LOSS` | 1.0 | `COORD_CONSITENCY_LOSS` | 1.0 | no change (keep prototype spelling) |
| (new) | — | `RANK_UNIFORM_LOSS` | 10_000.0 | new |
| `NEIGHBOR_LAMBDA` | 0.5 | `NEIGHBOR_LAMBDA` | 0.5 | no change |
| `REPULSION_LAMBDA` | 0.1 | `REPULSION_LAMBDA` | 0.1 | no change |

**Loss computation structure** also changes. The production training loop currently combines supervised and pseudo losses before weighting:
```python
# Current production
pred_loss = sup_loss + PSEUDO_PRED_LAMBDA * pseudo_loss
total_loss += PRED_LAMBDA * pred_loss
```
The prototype applies separate top-level weights to each, matching the table above:
```python
# Target (prototype structure)
total_loss += PRED_SUP_LAMBDA * sup_pred_loss + PRED_PSEUDO_LAMBDA * pseudo_pred_loss
```
The `total_loss` computation in `training.py` must be updated to match the prototype's structure exactly (with stale `simclr`/`SIMCLR_EMB_LOSS` variable names replaced by `swav`/`SWAV_EMB_LOSS`).

---

## 5. Label Weight Tracker — `LabeledRateTracker`

### 5.1 Differences from `losses.py` version

| Aspect | `losses.py` (production) | `utils.py` (prototype v4) |
|--------|--------------------------|---------------------------|
| `num_updates` counter | Not present | Present (line 185) — used for `NBATCH_PSEUDO_WARMUP` |
| `class_freq` attribute | Not exposed (counts returned from `update()` only) | Separate `class_freq` tensor stored on the object (lines 228-229) — accessed directly as `label_tracker.class_freq` |
| `class_weights` sync | Updated in-place | Copied from `class_freq` via `.copy_()` |
| Dead class handling | `store[store < 1.0/total] = 0.0` | `store[store < 1e-5] = 0.0` (line 208) |
| `get_class_weights()` when empty | Returns `None` | Returns `torch.ones_like(store)` (line 250) |
| `share_memory_()` | Not used | Commented out (lines 182, 184) |

**Action**: Merge the following from the prototype into `losses.py`'s `LabeledRateTracker`:
- Add `num_updates` counter — incremented on each `update()` call, needed for `NBATCH_PSEUDO_WARMUP` guard (`start_v4_sqlite.py:386`: `label_tracker.num_updates > NBATCH_PSEUDO_WARMUP`)
- Add `class_freq` as a stored tensor attribute — the training loop accesses it directly via `sum(label_tracker.class_freq) > 0` (`start_v4_sqlite.py:386`)

**Note**: `AdaptiveThreshold` in the prototype is already set to CUDA device (not CPU). No device change needed for that class.

### 5.2 `AdaptiveThreshold` — add to losses module

**Action**: Add `AdaptiveThreshold` class from `ps_prototypes_v2/utils.py:1396` to `losses.py`.

**Purpose**: Per-class adaptive confidence thresholds for pseudo-label selection, replacing the fixed `PSEUDO_THRESH` threshold. Placing it in `losses.py` avoids an inverted import dependency (`losses.py` would otherwise need to import from `scoring.py` to implement `prediction_loss_pseudo_sce_adaptive`).

---

## 6. Critical Bug Fix: Labels from Different Views

### 6.1 The issue

The line:
```python
labels = labels.repeat(len(views))
```
at `start_v4_sqlite.py:362` repeats ground truth labels across all views. Combined with:
```python
proj_emb = proj_emb.view(-1, proj_emb.shape[-1])
proj_coords = proj_coords.view(-1, 2)
pred_logits = pred_logits.view(-1, pred_logits.shape[-1])
```

The flat layout after `view(-1)` is `[V*B, ...]` in **views-major** order: `[v0_b0, v0_b1, ..., v1_b0, v1_b1, ...]`.

The `labels.repeat(V)` produces `[l_b0, l_b1, ..., l_bN, l_b0, l_b1, ...]` matching this views-major layout.

### 6.2 What was broken in v3

In `start_v3.py:274`, `torch.stack(views_gpu, dim=1)` produces `[B, V, C, H, W]` which when flattened with `flatten(0, 1)` becomes `[B*V, ...]` in **samples-major** order: `[b0_v0, b0_v1, ..., b1_v0, b1_v1, ...]`.

This means `labels.repeat(V)` does NOT align with the flat tensor layout — labels for patch 0 appear at indices `0, V, 2V, ...` but the flat tensor has them at `0, 1, ...` for the first view.

### 6.3 Fix in v4

`torch.stack(views_gpu, dim=0)` produces `[V, B, C, H, W]` which when flattened with `flatten(0, 1)` becomes `[V*B, ...]` in **views-major** order, correctly aligned with `labels.repeat(V)`.

**Status**: This fix is already present in the production `losses.py` — `prediction_loss_pseudo` already assumes views-major layout `[v0_b0, v0_b1, ...]`. The integration task is to ensure the training loop in `training.py` uses `torch.stack(views_gpu, dim=0)` (not `dim=1`) when constructing the view tensor, so the layout assumption in `losses.py` remains satisfied.

---

## 7. Integration Checklist

### 7.1 Database changes
- [ ] Add `priority` column to `patch_model()` in `patchsorter/db/head_client/models.py`
- [ ] Add partial index on `priority` where `priority > 0` (SQL migration)

### 7.2 New classes to add to `patchsorter/dl/`
- [ ] `IterableShardDataset` — sequential shard iterator (place in `datasets.py`, from `ShardDataset` pattern)
- [ ] `EnrichedInfiniteIterableDataset` — enriched infinite dataloader (place in `datasets.py`, from `CandidatePoolIterableDataset`)
- [ ] `CandidatePool` — in-memory candidate pool with score decay (place in `datasets.py`, from `GTCandidatePool`)
- [ ] `AdaptiveThreshold` — per-class adaptive threshold (place in `losses.py`, from `utils.py:1396`)
- [ ] `ScoreWriter` — DB score writer using batched updates (place in `scoring.py`; calls the DB worker client but belongs in the DL layer, not the DB layer)

### 7.3 Loss functions to add/update in `losses.py`
- [ ] Rename `simclr_loss()` to `swav_loss()` and replace its NT-Xent implementation with the SwAV prototype-assignment computation from `ps_prototypes_v2/utils.py:1191`
- [ ] Add `AdaptiveThreshold` class (from `utils.py:1396`)
- [ ] Add `prediction_loss_pseudo_sce_adaptive()` (from `utils.py:1331`)
- [ ] Add `rank_uniform_loss()`
- [ ] Remove `max_mean_discrepancy()` call and `MAX_MEAN_LOSS` weight constant from `training.py` (function may remain in `losses.py` as dead code for now)
- [ ] Add `compute_weighting_scores()` to `patchsorter/dl/scoring.py`

### 7.4 `LabeledRateTracker` updates in `losses.py`
- [ ] Add `num_updates` counter (incremented each `update()` call)
- [ ] Add `class_freq` as a stored tensor attribute (required by training loop guard `sum(label_tracker.class_freq) > 0`)
- [ ] Update `get_class_weights()` to return uniform weights when empty (not None)
- [ ] Update dead class threshold from `1.0/total` to `1e-5`

### 7.5 Training loop changes in `training.py`
- [ ] Restructure `train_worker()` to support dual dataloader pattern
- [ ] Add `dataloader_sequential` iteration (until exhaustion)
- [ ] Add `dataloader_enriched` iteration (within sequential loop)
- [ ] GPU-side batch concatenation (preserving v4 pattern)
- [ ] Only sequential predictions saved
- [ ] Only base (sequential) batch labels update tracker — use `labels[0:nbase_ids]`
- [ ] Priority score update per batch on sequential part only — pass `nbase_ids` (not `len(ids)`) to `compute_weighting_scores()`
- [ ] Update loss computation to match prototype structure exactly: SwAV on embeddings only (remove `simclr_coord_loss` term), separate `PRED_SUP_LAMBDA`/`PRED_PSEUDO_LAMBDA` weights, split `SEMANTIC_LAMBDA`, drop MMD term
- [ ] Adaptive threshold for pseudo-labels (use `AdaptiveThreshold` instance; guard with `label_tracker.num_updates > NBATCH_PSEUDO_WARMUP`)
- [ ] Normalize `joint_head.prototypes.data` after each optimizer step

### 7.6 Config migration
- [ ] Apply all constant renames, splits, and value changes per the table in §4.4
- [ ] Split `SEMANTIC_LAMBDA` into `SEMANTIC_COORD_LAMBDA = 1.0` and `SEMANTIC_EMB_LAMBDA = 10.0`
- [ ] Rename `SIMCLR_EMB_LOSS` → `SWAV_EMB_LOSS`
- [ ] Rename `PRED_LAMBDA` → `PRED_SUP_LAMBDA` and update value from `100.0` to `10_000.0`
- [ ] Replace `PSEUDO_PRED_LAMBDA = 0.4` with `PSEUDO_PRED_LAMBDA = 0.0001` and add derived `PRED_PSEUDO_LAMBDA = PRED_SUP_LAMBDA * PSEUDO_PRED_LAMBDA`
- [ ] Set `COORD_CONTRASTIVE_LOSS = 0` (currently `1.0` and active in production)
- [ ] Remove `MAX_MEAN_LOSS`
- [ ] Add `RANK_UNIFORM_LOSS = 10_000.0`
- [ ] Add SwAV hyperparameters: `SWAV_PROTOTYPES=300`, `SWAV_KMEANS_ITERS=10`, `SWAV_SINKHORN_ITERS=3`, `SWAV_EPS=0.05`
- [ ] Add `NBATCH_PSEUDO_WARMUP` constant
- [ ] Add `GT_*` constants for candidate pool sizing
- [ ] Add `K_NEIGHBORS`, `GT_SPATIAL_RARITY_ALPHA`, `GT_CLASS_RARITY_ALPHA`

### 7.7 Augmentation integration
- [ ] Verify `augmentations.py` is compatible with prototype's `get_transforms()`

---

## 8. Open Questions — Resolved

1. **Priority column vs score_timestamp**: ✅ `priority` replaces `score_timestamp`.
2. **Partial index scope**: ✅ Only on `priority` column where `priority > 0`. No index needed for ground truth label column.
3. **Backend for enriched dataloader**: ✅ Production uses PostgreSQL via worker client (not SQLite). SQLite is prototyping-only.
4. **SwAV prototypes source**: ✅ Add `prototypes` attribute to `JointHead` model. Copy `swav_loss()` from prototype as-is; keep `SWAV_KMEANS_ITERS` as variable.
5. **`compute_weighting_scores` integration**: ✅ Computed per-batch on the sequential part of the batch.
6. **AdaptiveThreshold device**: ✅ Already set to CUDA in prototype. No change needed.
7. **ScoreWriter for PostgreSQL**: ✅ Place in `patchsorter/dl/scoring.py` using batched updates (not COPY). Belongs in the DL layer (calls the DB worker client); placing it in the DB layer would mix ML training concerns into the data access layer.
8. **`COORD_CONTRASTIVE_LOSS`**: ✅ Set to 0. Matching prototype loss computation is the priority.
9. **`SIMCLR_EMB_LOSS` rename**: ✅ Rename to `SWAV_EMB_LOSS`.
10. **`PRED_SUP_LAMBDA` weight**: ✅ Confirmed at 10,000.
11. **Priority column migration**: ✅ In this development instance, the database will be recreated after the SQLAlchemy schema is updated. No ALTER TABLE migration needed.
12. **Enriched dataloader sharding**: ✅ Query across shards via manual UNION on PostgreSQL (not Citus), with `ORDER BY priority` + `LIMIT`.
13. **Loss weight constants**: ✅ All constants from `configs.py` are preserved as-is — they have been tuned in the prototype.
14. **New constants (`NBATCH_PSEUDO_WARMUP`, `GT_*`, `K_NEIGHBORS`)**: ✅ Values preserved from prototype — carefully tuned and should not be modified.

---

## 9. File Structure After Integration

```
patchsorter/dl/
├── __init__.py
├── augmentations.py          # existing
├── losses.py                 # updated: swav_loss (replaces simclr_loss), AdaptiveThreshold, adaptive pseudo, rank_uniform, tracker
├── model.py                  # updated: JointHead.prototypes added
├── training.py               # updated: dual dataloader loop, constants updated to match prototype, MMD/coord-contrastive removed
├── datasets.py               # NEW: IterableShardDataset, EnrichedInfiniteIterableDataset, CandidatePool
└── scoring.py                # NEW: compute_weighting_scores, ScoreWriter
```
