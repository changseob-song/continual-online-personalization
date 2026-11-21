"""Task-free reservoir replay for the controller branch.

One replay unit is one unmodified adaptation segment (currently two gait
cycles), with its heel-strike indices for supervision. Capacity is PER LIMB;
max_replay_count is the number of historical segments retrieved per update.
The first four adaptation messages are used only for current-data training,
matching OnlineAdaptator_PC's storage warm-up. Reservoir sampling is uniform
over the subsequent valid segments, separately for each limb.

Retrieval (ER-reservoir): every stored segment is scored with the current
model exactly as OnlineAdaptator_PC scores its bins (gait-phase RMSE %, every
10th window, no gradient). Only segments with 2.5 < RMSE < 20 are eligible,
and up to max_replay_count of them are chosen: uniformly at random by default,
or, with error_sorting=True, highest RMSE first (PC's top-k rule). Training
steps use PC's loss gate (MSE < 10). Algorithm R itself only maintains the
buffer and plays no role in retrieval.
"""

from collections import Counter
import json
import multiprocessing as mp
import os
import pickle
from queue import Empty
import random
import re
import signal
import time
import traceback

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from Hyperparam import hyperparam_config
from Model import TCN

# Same values as OnlineAdaptator_PC (replay_threshold_lower/upper, loss_threshold).
REPLAY_RMSE_LOWER = 2.5   # % gait phase
REPLAY_RMSE_UPPER = 20.0  # % gait phase
LOSS_THRESHOLD = 10.0     # a batch takes an optimizer step only if MSE < this
PRINT_RESERVOIR_LOG = True  # per-update terminal log of replay picks and storage/replacement


class ReservoirBuffer:
    """Algorithm R storage and independent random replay retrieval.

    seen_count counts eligible incoming units, including discarded ones.
    Separate RNGs keep retrieval frequency from changing storage decisions.
    """

    def __init__(self, capacity, storage_seed=None, replay_seed=None):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("buffer_capacity must be a positive integer")
        self.capacity = capacity
        self.seen_count = 0
        self.entries = []
        self.storage_rng = random.Random(storage_seed)
        self.replay_rng = random.Random(replay_seed)

    def add(self, item):
        """Return the assigned slot, or None if the new unit is discarded."""
        self.seen_count += 1
        if len(self.entries) < self.capacity:
            self.entries.append(item)
            return len(self.entries) - 1
        slot = self.storage_rng.randrange(self.seen_count)
        if slot < self.capacity:
            self.entries[slot] = item
            return slot
        return None

    def sample_indices(self, count, candidates):
        """Uniformly pick up to `count` slots from `candidates` (fewer if short)."""
        return self.replay_rng.sample(list(candidates), min(count, len(candidates)))

    def state_dict(self):
        return {
            'capacity': self.capacity,
            'seen_count': self.seen_count,
            'entries': self.entries,
            'storage_rng_state': self.storage_rng.getstate(),
            'replay_rng_state': self.replay_rng.getstate(),
        }

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity:
            raise ValueError("Saved reservoir capacity differs from buffer_capacity; start a new run")
        count = state['seen_count']
        entries = state['entries']
        if not isinstance(count, int) or count < 0 or len(entries) != min(count, self.capacity):
            raise ValueError("Invalid reservoir checkpoint counts")
        self.seen_count = count
        self.entries = entries
        self.storage_rng.setstate(state['storage_rng_state'])
        self.replay_rng.setstate(state['replay_rng_state'])


class _NumpyCompatUnpickler(pickle.Unpickler):
    # Same NumPy compatibility rule as Utils, without importing TensorRT/GPIO.
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        elif module == 'numpy.multiarray':
            module = 'numpy.core.multiarray'
        return super().find_class(module, name)


class ReplayDataset(Dataset):
    """Same sliding windows, phase labels and normalization as PC.LoadData."""

    def __init__(self, input_data, mid_peak_idx, normalization, window_size, input_size):
        data = np.asarray(input_data)
        peaks = np.asarray(mid_peak_idx).reshape(-1)
        if data.ndim != 2 or data.shape[1] != input_size:
            raise ValueError("input_data must have shape (samples, input_size)")
        if len(data) < window_size or not np.isfinite(data).all():
            raise ValueError("Segment is too short for a TCN window or contains non-finite values")
        if (len(peaks) == 0 or not np.isfinite(peaks).all()
                or not np.equal(peaks, np.floor(peaks)).all()):
            raise ValueError("mid_peak_idx must contain integer heel-strike indices")
        peaks = peaks.astype(np.int64)
        bounds = np.r_[0, peaks, len(data)]
        if np.any(np.diff(bounds) <= 0):
            raise ValueError("Heel strikes must be ordered and strictly inside the segment")
        positions = np.arange(len(data))
        cycles = np.searchsorted(bounds, positions, side='right') - 1
        angle = ((positions - bounds[cycles]) / (bounds[cycles + 1] - bounds[cycles])) * (2 * np.pi)
        labels = np.column_stack((np.cos(angle), np.sin(angle)))
        input_mean, input_std, label_mean, label_std = normalization
        if np.any(input_std == 0) or np.any(label_std == 0):
            raise ValueError("Normalization standard deviations must be nonzero")
        self.input = (data - input_mean) / input_std
        self.label = (labels - label_mean) / label_std
        self.window_size = window_size

    def __len__(self):
        return len(self.input) - self.window_size + 1

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input[idx:idx + self.window_size], dtype=torch.float32).T,
            torch.tensor(self.label[idx + self.window_size - 1], dtype=torch.float32),
        )


def _task_str(label):
    return 'n/a' if label is None else '{}/{}'.format(*label)


def _reservoir_summary(memory):
    """Fill level, stream-id range and (incline/speed) composition of one limb's reservoir."""
    entries = memory.entries
    text = '{}/{} stored, {} seen'.format(len(entries), memory.capacity, memory.seen_count)
    if entries:
        ids = [entry['stream_id'] for entry in entries]
        counts = Counter(_task_str(entry.get('task_label')) for entry in entries)
        text += ' | ids #{}..#{} | {}'.format(
            min(ids), max(ids), '  '.join('{} x{}'.format(k, v) for k, v in sorted(counts.items())))
    return text


def _course_path(path, course_num):
    stem, extension = os.path.splitext(os.fspath(path))
    stem = re.sub(r'_\d+$', '', stem)
    return '{}_{}{}'.format(stem, course_num, extension)


def _phase_rmse(model, dataset, device):
    """Gait-phase RMSE (%) computed exactly like PC's rmse_monitoring on a bin.

    Every 10th window, batch 16, no gradient, current model (left in train
    mode, as in PC). The angle is taken from the normalized (x, y) outputs
    and labels, as PC does, and the error is wrapped into [-50, 50] %.
    """
    loader = DataLoader(Subset(dataset, range(0, len(dataset), 10)), batch_size=16,
                        shuffle=False, num_workers=0, pin_memory=False)
    labels, outputs = [], []
    with torch.no_grad():
        for inputs, label in loader:
            outputs.append(model(inputs.to(device)).cpu())
            labels.append(label)
    labels, outputs = torch.cat(labels, dim=0), torch.cat(outputs, dim=0)

    def to_percent(coords):
        angle = torch.atan2(coords[:, 1], coords[:, 0])
        return (angle + 2 * np.pi) % (2 * np.pi) / (2 * np.pi) * 100

    error = to_percent(outputs) - to_percent(labels)
    error[error < -50] += 100
    error[error > 50] -= 100
    return torch.sqrt(torch.mean(error ** 2)).item()


def _train_update(model, optimizer, dataset, device, adaptation_on, generator):
    """One shuffled pass, batch size 16, Adam on linear only, MSE < LOSS_THRESHOLD.

    Returns (optimizer_steps, avg_loss); avg_loss is the mean loss over the
    batches that took a step, or NaN if none did.
    """
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0,
                        pin_memory=(device == 'cuda'), generator=generator)
    criterion = torch.nn.MSELoss()
    optimizer_steps = 0
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss_value = loss.item()
        if adaptation_on and loss_value < LOSS_THRESHOLD:
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total_loss += loss_value
    avg_loss = total_loss / optimizer_steps if optimizer_steps else float('nan')
    return optimizer_steps, avg_loss


class OnlineAdaptator_reservoir:
    """Drop-in adaptor; the first eight constructor arguments match PC.

    PCA/encoder paths are accepted for interface compatibility only. Incline
    and speed are kept with each stored segment purely as a logging label
    (replayed_bins_task_label); they are never used for storage or selection.
    Set random_seed to an integer for repeatability, or None for a fresh run.
    replay_sample_budget optionally caps the number of replay TCN windows.
    It never pads a shortage with duplicated samples.
    error_sorting picks replay segments among the eligible ones: False = random,
    True = top-k highest RMSE (same rule as PC).
    """

    def __init__(self, model_path, pca_model_path, encoder_model_path, course_num,
                 linear_layer_path, buffer_file_path, adaptation_ON=False,
                 replay_buffer_ON=False, *,
                 buffer_capacity=hyperparam_config['reservoir_buffer_capacity'],
                 max_replay_count=4, random_seed=hyperparam_config['reservoir_seed'],
                 min_adaptation_before_replay=4, replay_sample_budget=None,
                 error_sorting=False):
        # Validate configuration in the parent, before starting the worker.
        ReservoirBuffer(buffer_capacity)
        if not isinstance(error_sorting, bool):
            raise ValueError('error_sorting must be True or False')
        for name, value in [('max_replay_count', max_replay_count),
                            ('min_adaptation_before_replay', min_adaptation_before_replay)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('{} must be a nonnegative integer'.format(name))
        if replay_sample_budget is not None and (
                isinstance(replay_sample_budget, bool)
                or not isinstance(replay_sample_budget, int) or replay_sample_budget < 0):
            raise ValueError('replay_sample_budget must be None or a nonnegative integer')
        if random_seed is not None and (isinstance(random_seed, bool) or not isinstance(random_seed, int)):
            raise ValueError('random_seed must be None or an integer')
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()
        self._stopped = False
        self.adaptation_process = mp.Process(
            target=_adaptation_worker_process,
            args=(self.input_q, self.output_q, model_path, course_num,
                  linear_layer_path, buffer_file_path, dict(hyperparam_config),
                  adaptation_ON, replay_buffer_ON, buffer_capacity, max_replay_count,
                  random_seed, min_adaptation_before_replay, replay_sample_budget,
                  error_sorting),
        )
        self.adaptation_process.start()

    def trigger_finetuning(self, side, incline, speed, input_data, mid_peak_idx, start_idx):
        if self._stopped:
            raise RuntimeError('The adaptation worker has stopped')
        self.input_q.put((side, incline, speed, input_data, mid_peak_idx, start_idx))

    def get_updated_weights(self):
        try:
            return self.output_q.get_nowait()
        except Empty:
            if self.adaptation_process.exitcode not in (None, 0):
                raise RuntimeError('Reservoir worker failed; see its error traceback')
            return None

    def save_buffer_to_file(self):
        # Only the worker owns the live buffer; asking it avoids stale parent state.
        if not self._stopped:
            self.input_q.put(('save',))

    def stop_worker(self):
        if self._stopped:
            return
        self._stopped = True
        self.input_q.put((None, None, None, None, None, None))
        deadline = time.monotonic() + 30
        # Drain output while joining: queued weight arrays can otherwise block
        # the worker's multiprocessing Queue feeder during process shutdown.
        while self.adaptation_process.is_alive() and time.monotonic() < deadline:
            try:
                self.output_q.get(timeout=0.05)
            except Empty:
                pass
            self.adaptation_process.join(timeout=0.05)
        if self.adaptation_process.is_alive():
            self.adaptation_process.terminate()
            self.adaptation_process.join()
            print('Reservoir worker timed out; final checkpoint may not have been saved', flush=True)
        exitcode = self.adaptation_process.exitcode
        self.input_q.close()
        self.input_q.cancel_join_thread()
        self.output_q.close()
        self.output_q.cancel_join_thread()
        if exitcode:
            raise RuntimeError('Reservoir worker exited with code {}'.format(exitcode))


def _adaptation_worker_process(input_q, output_q, model_path, course_num,
                               linear_layer_path, buffer_file_path, config,
                               adaptation_on, replay_on, capacity, replay_count,
                               random_seed, min_before_replay, sample_budget, error_sorting):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _run_worker(input_q, output_q, model_path, course_num, linear_layer_path,
                    buffer_file_path, config, adaptation_on, replay_on, capacity,
                    replay_count, random_seed, min_before_replay, sample_budget,
                    error_sorting)
    except Exception:
        traceback.print_exc()
        raise


def _run_worker(input_q, output_q, model_path, course_num, linear_layer_path,
                buffer_file_path, config, adaptation_on, replay_on, capacity,
                replay_count, random_seed, min_before_replay, sample_budget,
                error_sorting=False):
    selection = 'top-k' if error_sorting else 'random'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_path = os.path.dirname(model_path)
    normalization = tuple(np.load(os.path.join(base_path, name + '.npy'))
                          for name in ['input_mean', 'input_std', 'label_mean', 'label_std'])
    models, optimizers, memories = {}, {}, {}
    for offset, side in enumerate(['L', 'R']):
        model = TCN(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        for param in model.parameters():
            param.requires_grad = False
        for param in model.linear.parameters():
            param.requires_grad = True
        model.train()
        models[side] = model
        optimizers[side] = torch.optim.Adam(model.linear.parameters(), lr=config['init_lr'])
        storage_seed = None if random_seed is None else random_seed + 2 * offset
        replay_seed = None if random_seed is None else random_seed + 2 * offset + 1
        memories[side] = ReservoirBuffer(capacity, storage_seed, replay_seed)

    loader_generator = torch.Generator()
    if random_seed is None:
        loader_generator.seed()
    else:
        loader_generator.manual_seed(random_seed)
    adaptation_count = 0
    checkpoint_path = _course_path(buffer_file_path, course_num)
    linear_save_path = _course_path(linear_layer_path, course_num)
    metrics_path = os.path.splitext(checkpoint_path)[0] + '_metrics.jsonl'

    if course_num > 1:
        with open(buffer_file_path, 'rb') as stream:
            checkpoint = _NumpyCompatUnpickler(stream).load()
        if checkpoint.get('format') != 'reservoir_replay_v1':
            raise ValueError('Expected a reservoir checkpoint; manifold/task buffers cannot be resumed')
        for side in ['L', 'R']:
            memories[side].load_state_dict(checkpoint['memories'][side])
        adaptation_count = checkpoint['adaptation_count']
        loader_generator.set_state(checkpoint['loader_rng_state'])
        # Checkpoints saved before this option existed always used random selection.
        saved_selection = 'top-k' if checkpoint.get('error_sorting', False) else 'random'
        if saved_selection != selection:
            print('WARNING: reservoir replay selection changed within a subject: {} was saved with {}, '
                  'this course uses {}'.format(buffer_file_path, saved_selection, selection), flush=True)
        with open(linear_layer_path, 'rb') as stream:
            parameters = _NumpyCompatUnpickler(stream).load()
        for side in ['L', 'R']:
            with torch.no_grad():
                models[side].linear.weight.copy_(torch.as_tensor(parameters['weights_' + side], device=device))
                models[side].linear.bias.copy_(torch.as_tensor(parameters['biases_' + side], device=device))
        if PRINT_RESERVOIR_LOG:
            print('Reservoir restored from {}\n  L: {}\n  R: {}'.format(
                buffer_file_path, _reservoir_summary(memories['L']), _reservoir_summary(memories['R'])), flush=True)

    def make_dataset(data, mid):
        return ReplayDataset(data, mid, normalization, config['window_size'], config['input_size'])

    def save_checkpoint():
        parameters = {}
        for side in ['L', 'R']:
            parameters['weights_' + side] = models[side].linear.weight.detach().cpu().numpy().copy()
            parameters['biases_' + side] = models[side].linear.bias.detach().cpu().numpy().copy()
        checkpoint = {
            'format': 'reservoir_replay_v1',
            'replay_unit': 'adaptation_segment',
            'memories': {side: memory.state_dict() for side, memory in memories.items()},
            'adaptation_count': adaptation_count,
            'loader_rng_state': loader_generator.get_state(),
            'max_replay_count': replay_count,
            'replay_sample_budget': sample_budget,
            'error_sorting': error_sorting,
            'random_seed': random_seed,
            'replay_rmse_range': (REPLAY_RMSE_LOWER, REPLAY_RMSE_UPPER),
            'loss_threshold': LOSS_THRESHOLD,
        }
        # Atomic replacement avoids leaving a partial checkpoint on interruption.
        for path, state in [(checkpoint_path, checkpoint), (linear_save_path, parameters)]:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary_path = path + '.tmp'
            with open(temporary_path, 'wb') as stream:
                pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, path)
        print('Reservoir saved: {} (L {}, R {})'.format(
            checkpoint_path, len(memories['L'].entries), len(memories['R'].entries)), flush=True)
        if PRINT_RESERVOIR_LOG:
            print('  L: {}\n  R: {}'.format(
                _reservoir_summary(memories['L']), _reservoir_summary(memories['R'])), flush=True)

    # Warm up a disposable model so neither limb's starting weights are trained.
    dummy = TCN(config).to(device)
    dummy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    dummy_optimizer = torch.optim.Adam(dummy.linear.parameters(), lr=config['init_lr'])
    dummy_data = np.zeros((max(300, config['window_size'] + 1), config['input_size']), dtype=np.float32)
    warmup_dataset = make_dataset(dummy_data, [10])
    warmup_loader = DataLoader(warmup_dataset, batch_size=8, shuffle=True,
                              num_workers=0, generator=loader_generator)
    inputs, labels = next(iter(warmup_loader))
    dummy_optimizer.zero_grad()
    torch.nn.functional.mse_loss(dummy(inputs.to(device)), labels.to(device)).backward()
    dummy_optimizer.step()
    del dummy, dummy_optimizer, warmup_loader
    print('Reservoir worker ready on {}; capacity {} per limb, replay up to {} ({}), seed {}'.format(
        device, capacity, replay_count, selection, random_seed), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, 'a', buffering=1) as metrics_stream:
        while True:
            message = input_q.get()
            if message[0] is None:
                save_checkpoint()
                return
            if message[0] == 'save':
                save_checkpoint()
                continue
            start_time = time.perf_counter()
            side, incline, speed, input_data, mid_peak_idx, end_idx = message
            if side not in memories:
                print('Reservoir worker: skipping invalid limb {}'.format(side), flush=True)
                continue
            try:
                current_dataset = make_dataset(input_data, mid_peak_idx)
            except (TypeError, ValueError) as error:
                print('Reservoir worker: skipping invalid segment: {}'.format(error), flush=True)
                continue

            memory = memories[side]
            replay_ready = adaptation_count >= min_before_replay or course_num > 1
            selected, stored_datasets, stored_rmse, eligible = [], [], [], []
            if replay_on and replay_ready and memory.entries:
                # Score every stored segment with the current (pre-update) model,
                # keep those inside PC's RMSE band, then pick among them.
                stored_datasets = [make_dataset(entry['input_data'], entry['mid_peak_idx'])
                                   for entry in memory.entries]
                stored_rmse = [_phase_rmse(models[side], dataset, device) for dataset in stored_datasets]
                eligible = [i for i, rmse in enumerate(stored_rmse)
                            if REPLAY_RMSE_LOWER < rmse < REPLAY_RMSE_UPPER]
                if error_sorting:
                    # PC's rule: highest RMSE first, up to replay_count (stable sort: ties keep slot order).
                    selected = sorted(eligible, key=lambda i: stored_rmse[i], reverse=True)[:replay_count]
                else:
                    selected = memory.sample_indices(replay_count, eligible)
            historical_datasets = [stored_datasets[i] for i in selected]
            replay_ids = [memory.entries[i]['stream_id'] for i in selected]
            # Logging only, same order as replay_ids: (incline, speed) of each replayed segment.
            replay_task_labels = [memory.entries[i].get('task_label') for i in selected]
            historical_dataset = ConcatDataset(historical_datasets) if historical_datasets else None
            available_windows = len(historical_dataset) if historical_dataset is not None else 0
            if historical_dataset is not None and sample_budget is not None and available_windows > sample_budget:
                indices = memory.replay_rng.sample(range(available_windows), sample_budget)
                historical_dataset = Subset(historical_dataset, indices)
            replay_windows = len(historical_dataset) if historical_dataset is not None else 0
            datasets = ([historical_dataset] if replay_windows else []) + [current_dataset]
            combined_dataset = ConcatDataset(datasets)
            optimizer_steps, avg_loss = _train_update(models[side], optimizers[side], combined_dataset,
                                                      device, adaptation_on, loader_generator)

            # Insert AFTER retrieval/training so current data are not replayed
            # a second time in the very update in which they first arrive.
            retained_slot = None
            # Snapshot for the log only: which stored segment a replacement evicts.
            was_full = len(memory.entries) >= memory.capacity
            previous_entries = list(memory.entries)
            if replay_ready:
                retained_slot = memory.add({
                    'input_data': np.array(input_data, copy=True),
                    'mid_peak_idx': np.asarray(mid_peak_idx, dtype=np.int64).reshape(-1).copy(),
                    'stream_id': memory.seen_count + 1,
                    'task_label': (incline, speed),  # logging only; never used for storage/selection
                })
            adaptation_count += 1
            duration = time.perf_counter() - start_time
            metrics_stream.write(json.dumps({
                'side': side, 'end_idx': int(end_idx), 'update': adaptation_count,
                'buffer_size': len(memory.entries), 'seen_count': memory.seen_count,
                'retained_slot': retained_slot, 'replay_ids': replay_ids,
                'replay_selection': selection,
                'replay_units': len(selected), 'current_windows': len(current_dataset),
                'available_replay_windows': available_windows, 'replay_windows': replay_windows,
                'total_windows': len(combined_dataset), 'optimizer_steps': optimizer_steps,
                'avg_loss': avg_loss if optimizer_steps else None,
                'batch_size': 16, 'duration_s': duration,
            }) + '\n')
            model = models[side]
            # Controller's ten-field message:
            # (side, start_index, update_dur, input_reduced, grid_key,
            #  replayed_bins, replayed_bins_task_label, avg_loss, weights, biases)
            # No PCA coordinates or grid keys exist here, so those get NaN / ()
            # placeholders. grid_key must stay iterable because Utils.save_data
            # calls tuple() on it. replayed_bins holds stream ids, and
            # replayed_bins_task_label the matching (incline, speed) per id.
            output_q.put((side, end_idx, duration, np.full(3, np.nan), (), replay_ids,
                          replay_task_labels,
                          avg_loss,
                          model.linear.weight.detach().cpu().numpy().copy(),
                          model.linear.bias.detach().cpu().numpy().copy()))

            if PRINT_RESERVOIR_LOG:
                # Printed after the weights are queued so logging never delays an update.
                task = _task_str((incline, speed))
                if not replay_on:
                    replay_line = 'off'
                elif not replay_ready:
                    replay_line = 'not yet (warm-up)'
                elif not stored_rmse:
                    replay_line = 'buffer empty'
                else:
                    # replay_ids/labels were taken before add(), which may have overwritten a picked slot.
                    picks = ', '.join('#{} {} {:.1f}%'.format(stream_id, _task_str(label), stored_rmse[i])
                                      for stream_id, label, i in zip(replay_ids, replay_task_labels, selected))
                    replay_line = 'scored {}, eligible {} ({}<RMSE<{}), picked {} ({}){}'.format(
                        len(stored_rmse), len(eligible), REPLAY_RMSE_LOWER, REPLAY_RMSE_UPPER,
                        len(selected), selection, ': ' + picks if picks else '')
                if not replay_ready:
                    store_line = 'warm-up update {}/{}, not stored'.format(adaptation_count, min_before_replay)
                else:
                    new_id = memory.seen_count
                    keep_p = '{}/{}={:.2f}'.format(memory.capacity, new_id, min(1.0, memory.capacity / new_id))
                    if retained_slot is None:
                        store_line = '#{} ({}) discarded  [keep p={}]'.format(new_id, task, keep_p)
                    elif was_full:
                        old = previous_entries[retained_slot]
                        store_line = '#{} ({}) -> slot {}, replaced #{} ({})  [keep p={}]'.format(
                            new_id, task, retained_slot, old['stream_id'],
                            _task_str(old.get('task_label')), keep_p)
                    else:
                        store_line = '#{} ({}) -> slot {}  [filling {}/{}]'.format(
                            new_id, task, retained_slot, len(memory.entries), memory.capacity)
                print('[Reservoir {}] update {} (end_idx {}) | task {}\n'
                      '  replay : {}\n'
                      '  store  : {}'.format(
                          side, adaptation_count, end_idx, task, replay_line, store_line), flush=True)
