# ER-reservoir baseline (`OnlineAdaptator_reservoir.py`)

Baseline for comparison against the PC-based manifold-aware replay
(`OnlineAdaptator_PC.py`). It plugs into the same controller and uses the same
model, training step, warm-up and replay error threshold. The **only intended
difference** is how replay memory is kept and chosen.

> **ER-reservoir**: Algorithm R maintains the buffer (it only decides which
> incoming segment to keep and which stored one to replace). Replay segments are
> drawn at random from stored segments that satisfy the same error threshold as
> the PC framework, and the model is trained on current + replayed data.

---

## 1. How to run

**`Main.py`**

```python
adaptation_ON    = True
replay_buffer_ON = True    # False = train on current data only (no replay)
PC_USE           = False
RESERVOIR_USE    = True    # selects OnlineAdaptator_reservoir; do not set together with PC_USE
```

`Controller` raises an error if both `PC_USE` and `RESERVOIR_USE` are True. If both are False, the task-based
adaptor (`OnlineAdaptator_task`) is used, as before.

**`Hyperparam.py` → `hyperparam_config`**

| Key | Default | Meaning |
|---|---|---|
| `reservoir_seed` | `42` | Random seed. **Use a different seed for each subject**, spaced ≥ 4 apart (e.g. subject no. × 100). |
| `reservoir_buffer_capacity` | `50` | Segments stored **per limb**. |

Rules:
- **Seed:** only course 1 uses it. Courses 2+ restore the RNG state from the previous course's checkpoint, so
  changing the seed partway through a subject has no effect.
- **Why ≥ 4 apart:** internally the code uses `seed`, `seed+1`, `seed+2` and `seed+3`. These are the
  storage and replay RNGs for L and R.
- **Capacity:** must stay the same for all courses of a subject. Loading a checkpoint saved with a different
  capacity raises an error.
- **Start each subject at course 1.** PC/task buffer files (`buffer_*.pkl` with `bin_*` keys) cannot be
  loaded; the worker raises an error.

---

## 2. What happens in one update

Each update is triggered by one adaptation segment (two gait cycles plus the heel-strike index) from one limb.

```
segment (side, incline, speed, input_data, mid_peak_idx, end_idx)
  │
  ├─ validate → skip if malformed / shorter than window / NaN / bad heel strikes
  │
  ├─ if replay ON and warm-up passed and buffer for this limb not empty:
  │     score EVERY stored segment with the current (pre-update) model
  │       = gait-phase RMSE %, every 10th window, no_grad      (same as PC)
  │     eligible = { segments with 2.5 < RMSE < 20 }           (same band as PC)
  │     replay   = random sample of up to 4 from eligible      (PC: top-RMSE first)
  │                (fewer than 4 eligible → replay fewer; none → no replay)
  │
  ├─ train linear layer: one shuffled pass over replay + current,
  │     batch 16, Adam (init_lr), MSE; a batch steps only if loss < 10   (same as PC)
  │
  ├─ if warm-up passed: Algorithm R add(current segment)
  │     buffer not full → append
  │     full → j = randrange(seen_count); if j < capacity: replace slot j, else discard
  │
  └─ send updated weights to Controller; append one line to metrics file
```

Notes:
- **Warm-up:** the first 4 adaptation messages (L and R combined) only train on current data and are not
  stored, as in PC. From course 2 on, storage and replay are active from the start.
- **Insert after training:** the current segment is added to the buffer *after* training. It is therefore
  never replayed in the same update it arrives in. This is the standard ER order.
- **Separate buffers:** L and R each have their own buffer, model, optimizer and RNGs.
- **Not used for selection:** incline and speed are stored with each segment as a logging label only. They
  never affect storage or retrieval.

---

## 3. Comparison with PC-based manifold-aware

| | PC (manifold-aware) | ER-reservoir |
|---|---|---|
| Storage unit | One segment per PCA grid cell `(PC1//2, PC2//2, cadence//10)`. A new segment in the same cell overwrites the old one. | One segment per slot, kept by Algorithm R (uniform over all segments seen) |
| Capacity | Unbounded (one slot per visited cell) | `reservoir_buffer_capacity` per limb (default 50) |
| Replay candidates | Stored bins with 2.5 < RMSE < 20 | **Same** |
| Replay choice | Highest-RMSE bins first, up to 4 | **Random** among candidates, up to 4 |
| RMSE scoring | Every 10th window, current model, `no_grad` | **Same** (verified identical to 6 decimals) |
| Training | Linear layer only, Adam, batch 16, one pass, loss gate < 10 | **Same** |
| Warm-up before storing | 4 messages | **Same** |
| Current segment in its own replay | Possible (stored before scoring) | No (stored after training) |
| Uses PCA / task labels | PCA for grid keys | No |

Both methods also behave the same way in two details:
- **Dropout stays on during scoring.** The model stays in `train()` mode throughout, so dropout (0.05) is active
  during both RMSE scoring and training.
- **Angle from normalized values.** The phase angle is taken from the *normalized* (x, y) outputs. With the
  current `label_mean ≈ 0` and `label_std ≈ (0.70711, 0.70710)` this makes no numerical difference (measured
  ≤ 7e-5 %).

---

## 4. Output files

Paths follow `Main.py`: course N loads `*_{N-1}.pkl` and writes `*_{N}.pkl` in the controller folder.

### `buffer_{N}.pkl` (checkpoint)
```python
{
  'format': 'reservoir_replay_v1',
  'memories': {'L': {...}, 'R': {...}},   # capacity, seen_count, entries, storage/replay RNG states
  'adaptation_count', 'loader_rng_state', 'max_replay_count', 'replay_sample_budget',
  'random_seed', 'replay_rmse_range': (2.5, 20.0), 'loss_threshold': 10.0,
}
# each entry: {'input_data', 'mid_peak_idx', 'stream_id', 'task_label': (incline, speed)}
```
Files are written atomically (`.tmp` → `os.replace`) when the worker stops.

### `linear_layer_weights_biases_{N}.pkl`
Final linear weights/biases for L and R. The Controller writes the same file at trial end, then the worker
overwrites it when it stops. The values are the same or slightly newer.

### `buffer_{N}_metrics.jsonl` (one line per update)
`side, end_idx, update, buffer_size, seen_count, retained_slot, replay_ids, replay_units, current_windows,
available_replay_windows, replay_windows, total_windows, optimizer_steps, avg_loss, batch_size, duration_s`

- `retained_slot`: the slot the current segment went into. `null` means the segment was discarded, or that
  warm-up was still running.
- `avg_loss`: `null` when no batch passed the loss gate.

### Trial CSVs (`{trial}-pc_R.csv`, `{trial}-pc_L.csv`)
Same columns as PC runs. For reservoir runs they contain:

| Column | Reservoir content |
|---|---|
| `update_dur_*` | Worker time for the update (s) |
| `input_reduced_*` | `[nan nan nan]` (no PCA) |
| `grid_key_*` | `()` (no grid) |
| `replayed_bins_*` | **stream IDs** of replayed segments, e.g. `[17, 3, 22]` |
| `replayed_bins_task_label_*` | `(incline, speed)` of each replayed segment, **same order**, e.g. `[(0, 1.2), (10, 0.8), (-10, 1.0)]` |

How to read these columns:
- **Stream ID:** the arrival number of a segment among stored-eligible segments. It is counted **separately per
  limb** (L and R both start at 1), so use `(side, stream_id)` as the key.
- **Stream ID vs. PC bin index:** a stream ID always refers to the same segment. A PC bin index refers to a grid
  cell whose content (and label) can be overwritten over time.
- **Parsing a cell:** use `ast.literal_eval(cell)`.

---

## 5. Changes to other files

| File | Change |
|---|---|
| `Controller.py` | Imports `OnlineAdaptator_reservoir`; new `RESERVOIR_USE` argument; error if combined with `PC_USE`. |
| `Main.py` | New `RESERVOIR_USE` flag passed to `Controller`. |
| `Hyperparam.py` | Added `reservoir_seed`, `reservoir_buffer_capacity`. |
| `Utils.py` (`save_data`) | `replayed_bins_*` / `replayed_bins_task_label_*` are written from the raw Python lists. Before, a trial in which every update replayed the same number of bins produced multi-line numpy arrays in each cell, with incline turned into a float (`0.`). This also applies to PC/task CSVs, which now have a consistent format. |

The worker sends the Controller's 10-field message:
`(side, start_index, update_dur, input_reduced, grid_key, replayed_bins, replayed_bins_task_label, avg_loss, weights, biases)`.

---

## 6. Verification (Jetson, 2026-11-20)

| Check | Result |
|---|---|
| RMSE scoring vs PC `rmse_monitoring` on the same segments | Identical (e.g. 38.754383 vs 38.754383); datasets identical |
| End-to-end worker on 48 real segments (`buffer_8.pkl`), CSV saved via real `save_data` | 48/48 updates; 44/44 stored labels match source; 48/48 CSV rows have matching id/label pairs |
| Replacement with capacity 5 (22 replaced, 12 discarded) | All 116 replayed (id, label) pairs were in the buffer at that time with the correct label; every chosen slot matched an independent Algorithm R re-simulation; final buffer data matches its labels |
| Algorithm R uniformity (50 items, capacity 5, 20,000 runs) | Inclusion probability 0.096–0.104 (expected 0.100) |
| Dropout effect on scoring (48 real segments, 30 repeats) | RMSE std median 0.16 %, max 0.88 %; 1/48 segments near a threshold changed eligibility |

---

## 7. Caveats

- **Update time grows with buffer size.** Every stored segment is scored each update: about 0.1 s with an
  empty buffer and about 0.8–1.0 s at 22 segments per limb. Expect roughly 1.5–2 s at 50. PC also scores
  all of its bins.
- **Metrics files are appended to.** `buffer_{N}_metrics.jsonl` is never replaced, and its name does not include
  the subject. Move or rename it (and the `buffer_*.pkl` files) after each subject.
- **Repeated `8_x` runs share a starting point.** Each one loads `buffer_7.pkl` and overwrites `buffer_8.pkl`, so
  every `8_x` starts from the course-7 state.
- **Don't run GPU jobs alongside the controller** on the Jetson. They compete with the real-time loop.
  GPU memory errors (`NvMap ... error 12`) have been seen on this board.
