# %%
from Hyperparam import hyperparam_config
from Model import TCN
import torch, os, time, random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.signal import find_peaks
from Utils import cartesian_to_percentage_tensor, NumpyCompatUnpickler
import pickle, signal

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

class OnlineAdaptator_task():
    def __init__(self, model_path, pca_model_path, encoder_path, course_num, linear_layer_path, buffer_file_path, adaptation_ON=False, replay_buffer_ON=False):
        self.model_path = model_path
        self.pca_model_path = pca_model_path
        self.course_num = course_num
        self.linear_layer_path = linear_layer_path
        self.buffer_file_path = buffer_file_path
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()
        self.adaptation_ON = adaptation_ON
        self.replay_buffer_ON = replay_buffer_ON

        self.incline_values = [-10, 0, 10]
        self.speed_values = [.2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        self.bin_number_per_task = 4

        self.bin_state = {inc: {spd: {side: {bin_idx: 0 for bin_idx in range(self.bin_number_per_task)} for side in ['L', 'R']} for spd in self.speed_values} for inc in self.incline_values}
        self.bin_data = {inc: {spd: {side: {bin_idx: None for bin_idx in range(self.bin_number_per_task)} for side in ['L', 'R']} for spd in self.speed_values} for inc in self.incline_values}
        self.bin_mid_idx = {inc: {spd: {side: {bin_idx: None for bin_idx in range(self.bin_number_per_task)} for side in ['L', 'R']} for spd in self.speed_values} for inc in self.incline_values}
        self.bin_loader = {inc: {spd: {side: {bin_idx: None for bin_idx in range(self.bin_number_per_task)} for side in ['L', 'R']} for spd in self.speed_values} for inc in self.incline_values}

        # Start the new standalone worker process
        self.adaptation_process = mp.Process(
            target=self.adaptation_worker_process,
            args=(self.input_q, self.output_q, self.model_path, self.pca_model_path, self.course_num, self.linear_layer_path, self.buffer_file_path, hyperparam_config, self.adaptation_ON, self.replay_buffer_ON)
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
        self.input_q.put((None, None, None, None, None, None))  # Send shutdown signal

        self.adaptation_process.join(timeout=1)
        if self.adaptation_process.is_alive():
            self.adaptation_process.terminate()
            self.adaptation_process.join()
    
    def save_buffer_to_file(self):
        try:
            # Save the buffer data to file
            buffer_file_name = self.buffer_file_path[:-5] + str(self.course_num) + '.pkl'
            with open(buffer_file_name, "wb") as f:
                pickle.dump({
                'bin_state': self.bin_state,
                'bin_data': self.bin_data,
                'bin_mid_idx': self.bin_mid_idx,
                }, f)
            print(f"Buffer saved to {buffer_file_name}... ", flush=True)
        except Exception as e:
            print(f"Error saving buffer data: {e}", flush=True)

    def adaptation_worker_warmup(self, model, optimizer, criterion, device, model_path, input_mean, input_std, label_mean, label_std):
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

    def adaptation_worker_process(self, input_q, output_q, model_path, pca_model_path, course_num, linear_model_path, buffer_file_path, hyperparam_config, adaptation_ON=False, replay_buffer_ON=False):
        """
        This worker process handles the fine-tuning of the model.
        All PyTorch and CUDA initializations happen inside this function.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN) # revents the worker from dying immediately when Ctrl+C is pressed,
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
        model_dummy = TCN(hyperparam_config).to(device) # Dummy model for warm-up
        model_L.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model_R.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model_dummy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # 2. Freeze layers and setup optimizers
        for model in [model_L, model_R]:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.linear.parameters():
                param.requires_grad = True
        
        # 3. Load the linear layer weights and biases from previous courses
        if (course_num > 1) and self.adaptation_ON:
            with open(linear_model_path, "rb") as f:
                linear_params = NumpyCompatUnpickler(f).load()
            model_L.linear.weight.data.copy_(torch.from_numpy(linear_params['weights_L']))
            model_L.linear.bias.data.copy_(torch.from_numpy(linear_params['biases_L']))
            model_R.linear.weight.data.copy_(torch.from_numpy(linear_params['weights_R']))
            model_R.linear.bias.data.copy_(torch.from_numpy(linear_params['biases_R']))
            print("Adaptation Worker: Loaded linear layer weights and biases from file.")
            with open(buffer_file_path, "rb") as f:
                buffer_data = NumpyCompatUnpickler(f).load()
            self.bin_state = buffer_data['bin_state']
            self.bin_data = buffer_data['bin_data']
            self.bin_mid_idx = buffer_data['bin_mid_idx']
            print("Adaptation Worker: Loaded replay buffer data from file.")
            
            print("Adaptation Worker: Reconstructing DataLoaders from saved buffer...")
            
            # Reconstruct DataLoaders
            self.bin_loader = {inc: {spd: {side: {bin_idx: None for bin_idx in range(self.bin_number_per_task)} for side in ['L', 'R']} for spd in self.speed_values} for inc in self.incline_values}

            for inc in self.incline_values:
                for spd in self.speed_values:
                    for side in ['L', 'R']:
                        for i in range(self.bin_number_per_task):
                            if self.bin_state[inc][spd][side][i] == 0:
                                continue # Skip empty bins
                            d_input = self.bin_data[inc][spd][side][i]
                            d_mid_peak = self.bin_mid_idx[inc][spd][side][i]

                            # Re-instantiate Dataset
                            dataset = LoadData(side, inc, spd, d_input, d_mid_peak, model_path, input_mean, input_std, label_mean, label_std)
                            
                            if dataset.initilized:
                                # Recreate the loader
                                # Using range based subset as seen in original code logic
                                train_indices = list(range(len(dataset)))
                                subset = Subset(dataset, train_indices)
                                loader = DataLoader(subset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
                                self.bin_loader[inc][spd][side][i] = loader
                            print(f"Reconstructed DataLoader for bin: {inc}-{spd}-{side}-{i}, Dataset size: {len(dataset)}")

            print("Adaptation Worker: Loader reconstruction complete.")

        optimizer_L = torch.optim.Adam(model_L.linear.parameters(), lr=hyperparam_config['init_lr'])
        optimizer_R = torch.optim.Adam(model_R.linear.parameters(), lr=hyperparam_config['init_lr'])
        optimizer_dummy = torch.optim.Adam(model_dummy.linear.parameters(), lr=hyperparam_config['init_lr']) # Dummy optimizer for warm-up
        criterion = torch.nn.MSELoss()

        model_L.train()
        model_R.train()
        model_dummy.train() # Dummy model for warm-up

        self.adaptation_worker_warmup(model_dummy, optimizer_dummy, criterion, device, model_path, input_mean, input_std, label_mean, label_std)

        avg_loss = 0
        misdetection_flag = False
        prev_input_data = None

        min_adaptation_before_replay = 4 # Minimum number of adaptation steps before starting replay
        min_adaptation_count = 0
        max_replay_num = 5 # Maximum number of bins to consider for replay
        loss_threshold = 1.0 # Loss threshold to accept a training step
        replay_threshold = 2.5 # RMSE threshold (%) to include a bin in the replay buffer

        # Main loop to wait for data and fine-tune
        while True:
            try:
                # Wait for data from the main controller
                msg  = input_q.get() # input data shape : (length, channel num)

                if msg[0] is None:  # Check for shutdown signal
                    print("Adaptation Worker: Shutdown signal received. Saving buffer and exiting...")
                    self.save_buffer_to_file()
                    break

                start_time = time.time()

                side, incline, speed, input_data, mid_peak_idx, start_idx = msg # input data shape : (length, channel num)
                if speed == 0: speed = 0.3

                misdetection_threshold = (3.5 - 1.5 * speed) * 100  # Convert to frames
                # Detect the heelstrike misdetection
                if (mid_peak_idx[0] > misdetection_threshold) or (input_data.shape[0] - mid_peak_idx[-1] > misdetection_threshold) or (np.diff(mid_peak_idx) > misdetection_threshold).any():
                    misdetection_flag = True
                    print(f"Adaptation Worker: Heelstrike misdetection detected {mid_peak_idx}, {input_data.shape[0]}.")
                else:
                    misdetection_flag = False

                # Determine either left or right side
                model = model_R if side == 'R' else model_L
                optimizer = optimizer_R if side == 'R' else optimizer_L

                bin_state_this_task = any(self.bin_state[incline][speed][side].values())
                print('bin state this task: ', bin_state_this_task)
                if bin_state_this_task and (not misdetection_flag):
                    input_data = input_data
                elif bin_state_this_task and misdetection_flag:
                    input_data = prev_input_data  # Use previous data
                elif not bin_state_this_task and misdetection_flag:
                    continue # Skip this iteration if no valid data is available
                    
                # Prepare data loader for the current task
                dataset = LoadData(side, incline, speed, input_data, mid_peak_idx, model_path, input_mean, input_std, label_mean, label_std)
                if not dataset.initilized: continue # Skip if dataset returns nothing

                train_indices = list(range(len(dataset)))
                subset = Subset(dataset, train_indices)
                train_loader_current_task = DataLoader(subset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True) # Use num_workers=0 to avoid potential multiprocessing issues within a multiprocessing worker

                false_idx = [idx for idx, state in self.bin_state[incline][speed][side].items() if not state]

                min_adaptation_passed = min_adaptation_count >= min_adaptation_before_replay
                if course_num > 1: min_adaptation_passed = True
                if min_adaptation_passed:
                    # If there is an empty bin, use it. Otherwise, shift the (k-1) latest bins and use the last one for the new data
                    if false_idx:
                        new_idx = false_idx[0] # Get the first available bin index
                        self.bin_state[incline][speed][side][new_idx] = 1
                        self.bin_data[incline][speed][side][new_idx] = input_data
                        self.bin_mid_idx[incline][speed][side][new_idx] = mid_peak_idx
                        self.bin_loader[incline][speed][side][new_idx] = train_loader_current_task

                    else:
                        for i in range(self.bin_number_per_task - 1):
                            self.bin_data[incline][speed][side][i] = self.bin_data[incline][speed][side][i + 1]
                            self.bin_mid_idx[incline][speed][side][i] = self.bin_mid_idx[incline][speed][side][i + 1]
                            self.bin_loader[incline][speed][side][i] = self.bin_loader[incline][speed][side][i + 1]

                        self.bin_data[incline][speed][side][self.bin_number_per_task - 1] = input_data
                        self.bin_mid_idx[incline][speed][side][self.bin_number_per_task - 1] = mid_peak_idx
                        self.bin_loader[incline][speed][side][self.bin_number_per_task - 1] = train_loader_current_task
                        
                    prev_input_data = input_data # Store current data to prepare misdetection cases

                # Buffer-related code starts here
                # Get the positive bins excluding the current task
                positive_bins = [(inc, spd, bin_idx) for inc in self.incline_values for spd in self.speed_values for bin_idx in range(self.bin_number_per_task) if (self.bin_state[inc][spd][side][bin_idx] > 0) and not (inc == incline and spd == speed)]
                # Aggregate all train loaders into a single list
                rmse_bins = {inc: {spd: {bin_idx: None for bin_idx in range(self.bin_number_per_task)} for spd in self.speed_values} for inc in self.incline_values}

                with torch.no_grad(): # Disable gradient calculation for inference

                    start_time_monitoring = time.time()
                    # inference on all positive bins to compute RMSE
                    for inc, spd, bin_idx in positive_bins:
                        # Skip the current task
                        if (inc == incline) and (spd == speed): continue

                        # Get the original subset from the loader
                        original_subset = self.bin_loader[inc][spd][side][bin_idx].dataset
                        
                        # Sample every 10th index from the original subset's indices
                        sampled_indices = original_subset.indices[::10]
                        sampled_subset = Subset(original_subset.dataset, sampled_indices)
                        
                        # Create a loader for the sampled dataset
                        full_loader = DataLoader(sampled_subset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

                        # Calculate RMSE
                        rmse_bins[inc][spd][bin_idx] = rmse_monitoring(model, full_loader, device)
                        # print(f"Adaptation Worker monitoring: {inc}-{spd}-{side}-{bin_idx}: {rmse_bins[inc][spd][bin_idx]:.2f} (RMSE)")
                    # print(f"Adaptation Worker: RMSE monitoring completed in {time.time() - start_time_monitoring:.2f} seconds.")

                # Select top-k highest RMSE bins
                k = min(max_replay_num, len(positive_bins))  # Choose up to 4
                top_k_bins = sorted(positive_bins, key=lambda x: rmse_bins[x[0]][x[1]][x[2]], reverse=True)[:k]
                print(f"Adaptation Worker: Top {k} bins: {top_k_bins} / {len(positive_bins)}")
                
                # Add bins to replay buffer only if their RMSE is above a threshold
                bins_for_replay = []
                train_loader_combined_list = []

                if replay_buffer_ON:
                    for inc, spd, bin_idx in top_k_bins:
                        if (replay_threshold < rmse_bins[inc][spd][bin_idx] < 15):
                            train_loader_combined_list.append(self.bin_loader[inc][spd][side][bin_idx])
                            bins_for_replay.append((inc, spd, bin_idx))
                
                # Add current task to the replay buffer if there is no misdetection
                if not misdetection_flag:
                    train_loader_combined_list.append(train_loader_current_task) # Add current task to the replay buffer only when there is no misdetection

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

                    print(f"Avg loss: {tloss / num_batches if num_batches > 0 else 'N/A'}")
                    min_adaptation_count += 1
                                    
                # After training, get the updated weights and send them back
                updated_weights = model.linear.weight.data.clone().cpu().numpy()
                updated_biases = model.linear.bias.data.clone().cpu().numpy()

                update_latency = time.time() - start_time

                output_q.put((side, start_idx, update_latency, 'None', 'None', 'None', updated_weights, updated_biases))

            except Exception as e:
                print(f"Adaptation worker error: {e}")
                # In case of an error, it's often better to break the loop
                break
        
        print("Adaptation Worker: Exiting.")


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