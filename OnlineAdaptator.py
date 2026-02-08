# %%
from Hyperparam import hyperparam_config
from Model import TCN
import torch, os, time, random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.signal import find_peaks
from Utils import cartesian_to_percentage_tensor, NumpyCompatUnpickler

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

def pca_transform_reconstruction(input_data, mid_peak_idx, pca_matrix, pca_mean, pca_scale):
    # Prepare the pca input
    if isinstance(mid_peak_idx, (np.ndarray, list)):
        mid_peak_idx = int(mid_peak_idx[0])
    
    input_length_1 = mid_peak_idx
    input_length_2 = input_data.shape[0] - mid_peak_idx # input data shape : (length, channel num)

    input_data_resampled_1 = upsampling_2d(input_data[:mid_peak_idx, :], 50)  # output data shape: (100, channel num)
    input_data_resampled_2 = upsampling_2d(input_data[mid_peak_idx:, :], 50)  # output data shape: (100, channel num)
    input_data_scaled_1 = (input_data_resampled_1.flatten() - pca_mean) / pca_scale # shape: (100,)
    input_data_scaled_2 = (input_data_resampled_2.flatten() - pca_mean) / pca_scale # shape: (100,)
    
    # Calculate PCA scores explicitly (without length appended)
    pca_scores_1 = np.dot(input_data_scaled_1, pca_matrix)
    pca_scores_2 = np.dot(input_data_scaled_2, pca_matrix)
    
    # Create the reduced vector for grid key (with length)
    input_data_reduced_1 = np.r_[pca_scores_1, input_length_1]
    input_data_reduced_2 = np.r_[pca_scores_2, input_length_2]

    # Reconstruct example data using ONLY the PCA scores
    input_data_reconstructed_scaled_1 = np.dot(pca_scores_1, pca_matrix.T)
    input_data_reconstructed_scaled_2 = np.dot(pca_scores_2, pca_matrix.T)
    
    input_data_reconstructed_1 = input_data_reconstructed_scaled_1 * pca_scale + pca_mean
    input_data_reconstructed_2 = input_data_reconstructed_scaled_2 * pca_scale + pca_mean

    # Calculate reconstruction error
    reconstruction_error_scaled_1 = np.sqrt(np.mean((input_data_scaled_1 - input_data_reconstructed_scaled_1) ** 2))
    reconstruction_error_scaled_2 = np.sqrt(np.mean((input_data_scaled_2 - input_data_reconstructed_scaled_2) ** 2))

    return np.mean([input_data_reduced_1, input_data_reduced_2], axis=0) , np.mean([reconstruction_error_scaled_1, reconstruction_error_scaled_2])

def adaptation_worker_process(input_q, output_q, model_path, pca_model_path, hyperparam_config, adaptation_ON=False, replay_buffer_ON=False):
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

    with open(pca_model_path, 'rb') as f:
        pca_file = NumpyCompatUnpickler(f).load()
    pca_matrix = pca_file['pca_matrix']  # shape: (original_dim, reduced_dim))
    pca_mean = pca_file['scaler_mean']      # shape: (original_dim,)
    pca_scale = pca_file['scaler_scale']    # shape: (original_dim,)

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

    bin_grid = {side: [] for side in ['L', 'R']} 
    bin_data = {side: [] for side in ['L', 'R']}
    bin_loader = {side: [] for side in ['L', 'R']}
    avg_loss = 0
    misdetection_flag = False
    
    min_adaptation_before_replay = 4 # Minimum number of adaptation steps before starting replay
    min_adaptation_count = 0
    min_replay_num = 4 # Minimum number of bins to consider for replay
    loss_threshold = 1.0 # Loss threshold to accept a training step
    replay_threshold = 3.0 # RMSE threshold (%) to include a bin in the replay buffer
    grid_resolution = 2
    cadence_resolution = 5

    # Main loop to wait for data and fine-tune
    while True:
        try:
            # Wait for data from the main controller
            side, incline, speed, input_data, mid_peak_idx, start_idx = input_q.get() # input data shape : (length, channel num)

            start_time = time.time()

            # Determine either left or right side
            model = model_R if side == 'R' else model_L
            optimizer = optimizer_R if side == 'R' else optimizer_L

            # 1. Go through PCA transformation
            input_data_reduced, reconstruction_error_scaled = pca_transform_reconstruction(input_data, mid_peak_idx, pca_matrix, pca_mean, pca_scale)
            # print(f"Adaptation Worker: PCA transformation done. Reconstruction error (scaled): {reconstruction_error_scaled:.2f}")

            # 2. Check misdetection based on reconstruction error 
            if reconstruction_error_scaled > 5.0:
                print(f"Adaptation Worker: High reconst. error detected: {reconstruction_error_scaled:.2f}")
                # misdetection_flag = True
            else:
                misdetection_flag = False
            
            if not misdetection_flag:

                # 3. Prepare the data loader for the current task
                dataset = LoadData(side, incline, speed, input_data, mid_peak_idx, model_path, input_mean, input_std, label_mean, label_std)
                if not dataset.initilized: continue # Skip if dataset returns nothing

                train_indices = list(range(len(dataset)))
                subset = Subset(dataset, train_indices)
                train_loader_current_task = DataLoader(subset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True) # Use num_workers=0 to avoid potential multiprocessing issues within a multiprocessing worker

                # 4. Get the coordinate of the reduced input data in replay buffer grid
                
                spatial = (input_data_reduced[:2] // grid_resolution).astype(int)   # PC parts (indices 0 and 1)
                temporal = int(input_data_reduced[2] // cadence_resolution)     # Temporal part (index 2) - cast to int explicitly
                grid_key = tuple(spatial.tolist()) + (temporal,)
                print("Grid key: ", grid_key, "Input reduced: ", input_data_reduced)

                if min_adaptation_count >= min_adaptation_before_replay:
                    if grid_key not in bin_grid[side]: # New grid point
                        bin_grid[side].append(grid_key)
                        bin_data[side].append(input_data)
                        bin_loader[side].append(train_loader_current_task)
                    else: # Existing grid point, update data
                        existing_idx = bin_grid[side].index(grid_key)
                        bin_data[side][existing_idx] = input_data
                        bin_loader[side][existing_idx] = train_loader_current_task


            # 6. Compute RMSE for all bins in the replay buffer
            with torch.no_grad(): # Disable gradient calculation for inference

                # inference on all positive bins to compute RMSE
                bin_rmse = []
                for bin_idx in range(len(bin_grid[side])):

                    # Get the original subset from the loader
                    original_subset = bin_loader[side][bin_idx].dataset
                    
                    # Sample every 10th index from the original subset's indices
                    sampled_indices = original_subset.indices[::10]
                    sampled_subset = Subset(original_subset.dataset, sampled_indices)
                    
                    # Create a loader for the sampled dataset
                    full_loader = DataLoader(sampled_subset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

                    # Calculate RMSE
                    bin_rmse.append(rmse_monitoring(model, full_loader, device))
                    # print(f"Adaptation Worker monitoring: {side} - {bin_idx}: {bin_rmse[bin_idx]:.2f} (RMSE)")
             
            # 7. Select top-k highest RMSE bins
            bin_size = len(bin_grid[side])
            k = min(min_replay_num, bin_size)  # Choose up to 4
            top_k_bins = sorted(range(bin_size), key=lambda x: bin_rmse[x], reverse=True)[:k]
            
            # 8. Add bins to replay buffer only if their RMSE is above a threshold
            bins_for_replay = []
            train_loader_combined_list = []

            if replay_buffer_ON:
                for bin_idx in top_k_bins:
                    if bin_rmse[bin_idx] > replay_threshold:
                        train_loader_combined_list.append(bin_loader[side][bin_idx])
                        bins_for_replay.append(bin_idx)
                print(f"Adaptation Worker: Replaying bins with RMSE > {replay_threshold}%: {bins_for_replay}")

            # Add the current task data to the replay if no misdetection
            if not misdetection_flag :
                train_loader_combined_list.append(train_loader_current_task)                    

            # 9. Combine all top k replay bins
            if train_loader_combined_list:
                train_loader_combined = torch.utils.data.ConcatDataset([loader.dataset for loader in train_loader_combined_list])
                train_loader_combined = DataLoader(train_loader_combined, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

                tloss = 0
                num_batches = 0

                for input_batch, label_batch in train_loader_combined:
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

                min_adaptation_count += 1
            
            avg_loss = tloss / num_batches if num_batches > 0 else 0
            # print(f"avg loss: {avg_loss:.3f}")
                    
            # After training, get the updated weights and send them back
            updated_weights = model.linear.weight.data.clone().cpu().numpy()
            updated_biases = model.linear.bias.data.clone().cpu().numpy()

            # print(f"time taken for adaptation: {time.time() - start_time:.2f} seconds")

            output_q.put((side, start_idx, input_data_reduced, grid_key, bins_for_replay, updated_weights, updated_biases))

        except Exception as e:
            print(f"Adaptation worker error: {e}")
            # In case of an error, it's often better to break the loop
            break
    
    print("Adaptation Worker: Exiting.")


class OnlineAdaptator():
    def __init__(self, model_path, pca_model_path, adaptation_ON=False, replay_buffer_ON=False):
        self.model_path = model_path
        self.pca_model_path = pca_model_path
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()
        self.adaptation_ON = adaptation_ON
        self.replay_buffer_ON = replay_buffer_ON

        # Start the new standalone worker process
        self.adaptation_process = mp.Process(
            target=adaptation_worker_process,
            args=(self.input_q, self.output_q, self.model_path, self.pca_model_path, hyperparam_config, self.adaptation_ON, self.replay_buffer_ON)
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