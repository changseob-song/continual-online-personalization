# %%
from Hyperparam import hyperparam_config
from Model import TCN
import torch, os, time, random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.signal import find_peaks
from Utils import cartesian_to_percentage_tensor

def adaptation_worker_warmup(model, optimizer, criterion, device, model_path, input_mean, input_std, label_mean, label_std):
    # --- Warm-up Phase ---
    print("Adaptation Worker: Starting warm-up...")
    try:
        # Create enough data for a few gait cycles and at least one batch
        dummy_input_data = np.zeros((300, hyperparam_config['input_size']), dtype=np.float32)

        # Use one of the models (e.g., model_R) for the warm-up
        warmup_dataset = LoadData('R', 'LG', '1p0mps', dummy_input_data, np.array([10]), model_path, input_mean, input_std, label_mean, label_std)
        if warmup_dataset.initilized and len(warmup_dataset) > 0:
            warmup_loader = DataLoader(warmup_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
            
            # Run one training step to initialize CUDA and DataLoader workers
            for input_batch, label_batch in warmup_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)
                optimizer.zero_grad()
                logits = model(input_batch)
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()
                break # Only need one step for warm-up
        print("Adaptation Worker: Warm-up complete.")

    except Exception as e:
        print(f"Adaptation Worker: Error during warm-up: {e}")

def upsampling(data, target_length):
    """
    Upsample a 1D numpy array to the desired target length using linear interpolation.
    """
    original_length = len(data)
    original_indices = np.linspace(0, original_length - 1, original_length)
    new_indices = np.linspace(0, original_length - 1, target_length)

    return np.interp(new_indices, original_indices, data)

def upsampling_2d(data, target_length):
    """
    Upsample a 2D numpy array to the desired target length using linear interpolation.
    data: numpy array of shape (n_features, n_samples)
    target_length: desired length of each segment after upsampling
    """
    upsampled_data = np.zeros((target_length, data.shape[1]))
    for i in range(data.shape[1]):
        upsampled_data[:, i] = upsampling(data[:, i], target_length)

    return upsampled_data  # (target_length, n_features)

def interpolate_two_cycles(cycle1, cycle2, weight):
    # cycle1 and cycle2 shape: (n_samples, n_features)
    if cycle2 is None:
        return cycle1
    interpolated_len = cycle1.shape[0]*weight + cycle2.shape[0]*(1-weight)
    cycle1_upsampled = upsampling_2d(cycle1, int(interpolated_len))
    cycle2_upsampled = upsampling_2d(cycle2, int(interpolated_len))
    interpolated_cycle = cycle1_upsampled * weight + cycle2_upsampled * (1 - weight)
    return interpolated_cycle

def rmse_monitoring(model, full_loader, device):
    labels_gp, outputs_gp = [], []
    for inputs, labels in full_loader:
        outputs = model(inputs.to(device))
        labels_gp.append(labels)
        outputs_gp.append(outputs.cpu())

    # Concatenate all batches
    all_labels = torch.cat(labels_gp, dim=0)
    all_outputs = torch.cat(outputs_gp, dim=0)
    labels_percent = cartesian_to_percentage_tensor(all_labels)
    outputs_percent = cartesian_to_percentage_tensor(all_outputs)

    # Vectorized error calculation
    error = outputs_percent - labels_percent
    error[error < -50] += 100
    error[error > 50] -= 100

    rmse = torch.sqrt(torch.mean(error**2)).item()
    return rmse

def adaptation_worker_process(input_q, output_q, model_path, hyperparam_config, adaptation_ON=False, replay_buffer_ON=False):
    """
    This worker process handles the fine-tuning of the model.
    All PyTorch and CUDA initializations happen inside this function.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Adaptation Worker: Using device: {device}")

    base_model_path = os.path.dirname(model_path)
    input_mean = np.load(os.path.join(base_model_path, 'input_mean.npy'))
    input_std = np.load(os.path.join(base_model_path, 'input_std.npy'))
    label_mean = np.load(os.path.join(base_model_path, 'label_mean.npy'))
    label_std = np.load(os.path.join(base_model_path, 'label_std.npy'))

    # 1. Initialize models inside the worker
    model_L = TCN(hyperparam_config).to(device)
    model_R = TCN(hyperparam_config).to(device)
    model_L.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model_R.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # 2. Freeze layers and setup optimizers
    for model in [model_L, model_R]:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.linear.parameters():
            param.requires_grad = True
    
    optimizer_L = torch.optim.Adam(model_L.linear.parameters(), lr=hyperparam_config['init_lr'])
    optimizer_R = torch.optim.Adam(model_R.linear.parameters(), lr=hyperparam_config['init_lr'])
    criterion = torch.nn.MSELoss()

    model_L.train()
    model_R.train()

    adaptation_worker_warmup(model_R, optimizer_R, criterion, device, model_path, input_mean, input_std, label_mean, label_std)

    incline_values = [-10, -5, 0, 5, 10]
    speed_values = [.3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    bin_state = {inc: {spd: {side: 0 for side in ['L', 'R']} for spd in speed_values} for inc in incline_values}
    input_stream = {inc: {spd: {side: None for side in ['L', 'R']} for spd in speed_values} for inc in incline_values}
    train_loader = {inc: {spd: {side: None for side in ['L', 'R']} for spd in speed_values} for inc in incline_values}
    avg_loss = 0
    misdetection_flag = False
    
    min_replay_num = 4 # Minimum number of bins to consider for replay
    loss_threshold = 1.0 # Loss threshold to accept a training step
    replay_threshold = 3.0 # RMSE threshold (%) to include a bin in the replay buffer

    # Main loop to wait for data and fine-tune
    while True:
        try:
            # Wait for data from the main controller
            side, incline, speed, input_data, mid_peak_idx, start_idx = input_q.get() # input data shape : (length, channel num)

            start_time = time.time()

            misdetection_threshold = (1.5 + 0.45 * speed) * 100  # Convert to frames
            # Detect the heelstrike misdetection
            if (mid_peak_idx[0] > misdetection_threshold) or (input_data.shape[0] - mid_peak_idx[-1] > misdetection_threshold) or (np.diff(mid_peak_idx) > misdetection_threshold).any():
                misdetection_flag = True
                print(f"Adaptation Worker: Heelstrike misdetection detected {mid_peak_idx}, {input_data.shape[0]}.")
            else:
                misdetection_flag = False

            # Determine either left or right side
            model = model_R if side == 'R' else model_L
            optimizer = optimizer_R if side == 'R' else optimizer_L

            # Interpolate with previous data if available
            if bin_state[incline][speed][side] == 1 and (not misdetection_flag):
                input_data = interpolate_two_cycles(input_data, input_stream[incline][speed][side], weight=0.5)
            elif bin_state[incline][speed][side] == 1 and misdetection_flag:
                input_data = prev_input_data  # Use previous data
            elif bin_state[incline][speed][side] == 0 and misdetection_flag:
                continue # Skip this iteration if no valid data is available
            
                
            # Prepare data loader for the current task
            dataset = LoadData(side, incline, speed, input_data, mid_peak_idx, model_path, input_mean, input_std, label_mean, label_std)
            if not dataset.initilized: continue # Skip if dataset returns nothing

            train_indices = list(range(len(dataset)))
            subset = Subset(dataset, train_indices)
            train_loader_current_task = DataLoader(subset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True) # Use num_workers=0 to avoid potential multiprocessing issues within a multiprocessing worker

            # Buffer-related code starts here
            # Get the positive bins excluding the current task
            positive_bins = [(inc, spd) for inc in incline_values for spd in speed_values if (bin_state[inc][spd][side] > 0) and not (inc == incline and spd == speed)]
            # Aggregate all train loaders into a single list
            rmse_bins = {inc: {spd: None for spd in speed_values} for inc in incline_values}

            with torch.no_grad(): # Disable gradient calculation for inference

                # inference on all positive bins to compute RMSE
                for inc, spd in positive_bins:
                    # Skip the current task
                    if (inc == incline) and (spd == speed): continue

                    # Get the original subset from the loader
                    original_subset = train_loader[inc][spd][side].dataset
                    
                    # Sample every 10th index from the original subset's indices
                    sampled_indices = original_subset.indices[::10]
                    sampled_subset = Subset(original_subset.dataset, sampled_indices)
                    
                    # Create a loader for the sampled dataset
                    full_loader = DataLoader(sampled_subset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

                    # Calculate RMSE
                    rmse_bins[inc][spd] = rmse_monitoring(model, full_loader, device)
                    print(f"Adaptation Worker monitoring: {inc}-{spd}-{side}: {rmse_bins[inc][spd]:.2f} (RMSE)")
                
            # Select top-k highest RMSE bins
            k = min(min_replay_num, len(positive_bins))  # Choose up to 4
            top_k_bins = sorted(positive_bins, key=lambda x: rmse_bins[x[0]][x[1]], reverse=True)[:k]
            
            # Add bins to replay buffer only if their RMSE is above a threshold
            bins_for_replay = []
            train_loader_combined_list = []

            for inc, spd in top_k_bins:
                if rmse_bins[inc][spd] > replay_threshold:
                    train_loader_combined_list.append(train_loader[inc][spd][side])
                    bins_for_replay.append((inc, spd))
                    
            # Combine all top k replay bins
            if replay_buffer_ON and train_loader_combined_list:
                print(f"Adaptation Worker: Replaying bins with RMSE > {replay_threshold}%: {bins_for_replay}")
                train_loader_combined = torch.utils.data.ConcatDataset([loader.dataset for loader in train_loader_combined_list])
                train_loader_combined = DataLoader(train_loader_combined, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

                # Training loop - other tasks that is already in the buffer
                for input_batch, label_batch in train_loader_combined:
                    input_batch = input_batch.to(device)
                    label_batch = label_batch.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(input_batch)
                    loss = criterion(logits, label_batch)

                    loss.backward()
                    optimizer.step()

            # Training loop - current task (put current task at the end to prioritize current task)
            tloss = 0
            num_batches = 0
            for input_batch, label_batch in train_loader_current_task:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)
                
                optimizer.zero_grad()
                logits = model(input_batch)
                loss = criterion(logits, label_batch)

                if adaptation_ON and (loss < loss_threshold):
                    loss.backward()
                    optimizer.step()
                    tloss += loss.item()
                    num_batches += 1
            
            # avg_loss = tloss / num_batches if num_batches > 0 else 0
            # print(f"avg loss: {avg_loss:.3f}")
                
            rmse_current = rmse_monitoring(model, train_loader_current_task, device)
                    
            if rmse_current < 10.0:
                bin_state[incline][speed][side] = 1
                input_stream[incline][speed][side] = input_data
                train_loader[incline][speed][side] = train_loader_current_task
                rmse_bins[incline][speed] = rmse_monitoring(model, train_loader_current_task, device)
                prev_input_data = input_data # Store current data to prepare misdetection cases
                print(f"Adaptation Worker monitoring: {incline}-{speed}-{side} (current): {rmse_bins[incline][speed]:.2f} (RMSE)")
                                
            # After training, get the updated weights and send them back
            updated_weights = model.linear.weight.data.clone().cpu().numpy()
            updated_biases = model.linear.bias.data.clone().cpu().numpy()

            print(f"time taken for adaptation: {time.time() - start_time:.2f} seconds")

            output_q.put((side, rmse_bins, rmse_current, start_idx, updated_weights, updated_biases))

        except Exception as e:
            print(f"Adaptation worker error: {e}")
            # In case of an error, it's often better to break the loop
            break
    
    print("Adaptation Worker: Exiting.")


class OnlineAdaptator():
    def __init__(self, model_path, adaptation_ON=False, replay_buffer_ON=False):
        self.model_path = model_path
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()
        self.adaptation_ON = adaptation_ON
        self.replay_buffer_ON = replay_buffer_ON

        # Start the new standalone worker process
        self.adaptation_process = mp.Process(
            target=adaptation_worker_process,
            args=(self.input_q, self.output_q, self.model_path, hyperparam_config, self.adaptation_ON, self.replay_buffer_ON)
        )
        self.adaptation_process.start()

    def trigger_finetuning(self, side, incline, speed, input_data, mid_peak_idx, start_idx):
        """Sends data to the adaptation worker to start fine-tuning."""
        self.input_q.put((side, incline, speed, input_data, mid_peak_idx, start_idx))

    def get_updated_weights(self):
        """Checks for and returns updated weights from the worker."""
        try:
            return self.output_q.get_nowait()
        except mp.queues.Empty:
            return None

    def stop_worker(self):
        """Sends a shutdown signal to the worker process."""
        self.input_q.put((None, None))
        self.adaptation_process.join() # Wait for the process to finish

class LoadData(Dataset):
    def __init__(self, side, incline, speed, input_data, mid_peak_idx, model_path, input_mean, input_std, label_mean, label_std, gait_cycle_index=None, num_gait_cycles=None):
        self.input = input_data
        self.window_size = hyperparam_config['window_size']
        self.model_path = model_path
        self.initilized = False

        peak_indices = mid_peak_idx.tolist()
        peak_indices.insert(0, 0)  # Add start index
        peak_indices.append(self.input.shape[0])  # Add end index
        print(f"{side}, inc: {incline}, spd: {speed}, HS idx: {peak_indices}")

        def gait_cycle_generator(peak_indices, num_cycles=None):
            # Create an array of all time indices
            all_indices = np.arange(peak_indices[0], peak_indices[-1])

            # Find which cycle each index belongs to
            cycle_indices = np.searchsorted(peak_indices, all_indices, side='right') - 1

            # Get the start and end of the cycle for each index
            cycle_starts = np.array(peak_indices)[cycle_indices]
            cycle_ends = np.array(peak_indices)[cycle_indices + 1]

            # Calculate normalized phase for all indices at once
            gc_polar_angle = (all_indices - cycle_starts) / (cycle_ends - cycle_starts) * 2 * np.pi
            
            # Calculate x and y coordinates
            gc_polar_x = np.cos(gc_polar_angle)
            gc_polar_y = np.sin(gc_polar_angle)
            
            return np.column_stack((gc_polar_x, gc_polar_y))

        # Calculate record time based on the length of the input data
        self.label = gait_cycle_generator(peak_indices)

        # Normalize input and label
        self.input = (self.input - input_mean) / input_std
        self.label = (self.label - label_mean) / label_std
        self.initilized = True

    def __len__(self):
        if not self.initilized:
            return 0
        return len(self.input) - self.window_size + 1

    def __getitem__(self, idx):
        windows_input = self.input[idx: idx + self.window_size]     # Shape: (window_size, input_size)

        # Convert to tensor without flattening
        window_input = torch.tensor(windows_input, dtype=torch.float32).T           # Shape: (input_size, window_size)

        # Get the target joint moments at the last time point in the window
        target_label = self.label[idx + self.window_size - 1]       # Shape: (output_size)
            
        window_label = torch.tensor(target_label, dtype=torch.float32) # Shape: (output_size), consider putting .T when output_size > 1
        
        return window_input, window_label