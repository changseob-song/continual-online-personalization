# %%
from Hyperparam import hyperparam_config
from Model import TCN
import torch, os, time
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.signal import find_peaks
from Utils import get_congruency_rmse_1d, get_congruency_rmse_2d, NumpyCompatUnpickler

def adaptation_worker_warmup(model, optimizer, criterion, device, model_path, input_mean, input_std, label_mean, label_std):
    # --- Warm-up Phase ---
    print("Adaptation Worker: Starting warm-up...")
    try:
        # Create enough data for a few gait cycles and at least one batch
        dummy_input_data = np.zeros((300, hyperparam_config['input_size']), dtype=np.float32)
        
        # Simulate peaks for gait cycle detection in LoadData
        dummy_input_data[50, 5] = -100
        dummy_input_data[150, 5] = -100
        dummy_input_data[250, 5] = -100

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
    upsampled_data = np.zeros((data.shape[0], target_length))
    for i in range(data.shape[0]):
        upsampled_data[i, :] = upsampling(data[i, :], target_length)

    return upsampled_data  # (target_length, n_features)

def interpolate_two_cycles(cycle1, cycle2, weight):
    # cycle1 and cycle2 shape: (n_features, n_samples)
    interpolated_len = cycle1.shape[1]*weight + cycle2.shape[1]*(1-weight)
    cycle1_upsampled = upsampling_2d(cycle1, int(interpolated_len))
    cycle2_upsampled = upsampling_2d(cycle2, int(interpolated_len))
    interpolated_cycle = cycle1_upsampled * weight + cycle2_upsampled * (1 - weight)
    return interpolated_cycle

def determine_task(input_data, mtr_pos_stream, mid_peak_idx, ab_avg_input, inclinations_all, speeds_all):
    combined_rmse_min = float('inf')
    for incline in inclinations_all:
        for speed in speeds_all:
            if incline != 'LG' and speed not in ['0p4mps', '0p5mps', '0p6mps', '0p7mps', '0p8mps', '0p9mps', '1p0mps']: continue
            if (incline == 'RD_10deg' or incline == 'RD_7p5deg') and speed in ['0p4mps', '0p5mps', '0p6mps', '0p7mps']: continue
            for gc in range(2):
                ab_avg_cycle = ab_avg_input[incline][speed][gc].T[:, 4]  # shape: (n_features, n_samples)
                # Extract the latest gait cycle from input_data
                if gc == 0:
                    start_idx = 0
                    end_idx = mid_peak_idx[0]
                elif gc == 1:
                    start_idx = mid_peak_idx[0]
                    end_idx = len(mtr_pos_stream)
                current_cycle = input_data[start_idx:end_idx][:, 4] # gyro_Y

                rmse, len_rmse = get_congruency_rmse_1d(current_cycle, ab_avg_cycle)
                combined_rmse = rmse + (len_rmse * 0.25)
                if combined_rmse < combined_rmse_min:
                    combined_rmse_min = combined_rmse
                    rmse_min = rmse
                    len_rmse_min = len_rmse
                    best_incline = incline
                    best_speed = speed
    
                # print(mid_peak_idx, len(mtr_pos_stream), len(current_cycle), len(ab_avg_cycle), rmse_min)
    print(f"RMSE: {rmse_min:.3f}, Length RMSE: {len_rmse_min}")
    return best_incline, best_speed, combined_rmse_min


def adaptation_worker_process(input_q, output_q, model_path, ab_avg_input_path, hyperparam_config, replay_buffer_ON=False):
    """
    This worker process handles the fine-tuning of the model.
    All PyTorch and CUDA initializations happen inside this function.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Adaptation Worker: Using device: {device}")

    with open(ab_avg_input_path, "rb") as f:
        ab_avg_input = NumpyCompatUnpickler(f).load()

    print(ab_avg_input['LG']['1p0mps'][0].shape)

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

    inclinations_all = ['RD_10deg', 'RD_5deg', 'LG', 'RA_5deg', 'RA_10deg']
    speeds_all = ['0p2mps', '0p4mps', '0p6mps', '0p8mps', '1p0mps', '1p2mps', '1p4mps']
    bin_state = {inc: {spd: 0 for spd in speeds_all} for inc in inclinations_all}
    input_stream = {inc: {spd: None for spd in speeds_all} for inc in inclinations_all}
    train_loader = {inc: {spd: None for spd in speeds_all} for inc in inclinations_all}

    # Main loop to wait for data and fine-tune
    while True:
        try:
            # Wait for data from the main controller
            side, input_data, mtr_pos_stream, mid_peak_idx = input_q.get() # input data shape : (length, channel num)
            
            start_time = time.time()

            # Determine either left or right side
            model = model_R if side == 'R' else model_L
            optimizer = optimizer_R if side == 'R' else optimizer_L

            # Determine the task by comparing congruency with AB average input
            incline, speed, rmse_min = determine_task(input_data, mtr_pos_stream, mid_peak_idx, ab_avg_input, inclinations_all, speeds_all)
            print(f"{time.time() - start_time:.3f}")
            print(f"\nDetected task - Incline: {incline}, Speed: {speed} with congruency RMSE: {rmse_min:.2f} for side {side}")

            # if bin_state[incline][speed] == 1:
            #     input_data = interpolate_two_cycles(input_data.T, input_stream[incline][speed].T, weight=0.5)
            #     input_data = input_data.T  # Transpose back to (length, channel num)

            # Prepare data loader for the current task
            dataset = LoadData(side, incline, speed, input_data, mid_peak_idx, model_path, input_mean, input_std, label_mean, label_std)
            if not dataset.initilized: continue # Skip if dataset returns nothing

            train_indices = list(range(len(dataset)))
            subset = Subset(dataset, train_indices)

            bin_state[incline][speed] = 1
            input_stream[incline][speed] = input_data
            train_loader[incline][speed] = DataLoader(subset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True) # !!!! Using num_workers=0 to avoid potential multiprocessing issues within a multiprocessing worker

            # Combine data from other tasks only if the bin is filled up
            train_loader_combined_list = []
            if replay_buffer_ON:
                # Aggregate all train loaders into a single list
                for inc in inclinations_all:
                    for spd in speeds_all:
                        if (inc == incline) and (spd == speed): continue # Skip the current task for separate backpropagation
                        if bin_state[inc][spd] == 1:
                            # if side == 'R':
                            #     print(f"Adaptation Worker: Including data from {inc}-{spd} for training. {len(train_loader[inc][spd].dataset)} samples.")
                            train_loader_combined_list.append(train_loader[inc][spd])

                train_loader_combined = torch.utils.data.ConcatDataset([loader.dataset for loader in train_loader_combined_list])
                if side == 'R':
                    print(f"Adaptation Worker: Total training samples combined: {len(train_loader_combined)}")
                train_loader_combined = DataLoader(train_loader_combined, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

            # Training loop - other tasks that is already in the buffer
            if train_loader_combined_list:
                for input_batch, label_batch in train_loader_combined:
                    input_batch = input_batch.to(device)
                    label_batch = label_batch.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(input_batch)
                    loss = criterion(logits, label_batch)

                    loss.backward()
                    optimizer.step()

            tloss = 0
            num_batches = 0
            # Training loop - current task
            for input_batch, label_batch in train_loader[incline][speed]:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)
                
                optimizer.zero_grad()
                logits = model(input_batch)
                loss = criterion(logits, label_batch)

                loss.backward()
                optimizer.step()
                tloss += loss.item()
                num_batches += 1

            avg_loss = tloss / num_batches if num_batches > 0 else 0
            print("avg loss: ", avg_loss)

            # After training, get the updated weights and send them back
            updated_weights = model.linear.weight.data.clone().cpu().numpy()
            updated_biases = model.linear.bias.data.clone().cpu().numpy()

            update_time = time.time() - start_time

            output_q.put((side, incline, speed, avg_loss, update_time, updated_weights, updated_biases))

        except Exception as e:
            print(f"Adaptation worker error: {e}")
            # In case of an error, it's often better to break the loop
            break
    
    print("Adaptation Worker: Exiting.")


class OnlineAdaptator():
    def __init__(self, model_path, ab_avg_input_path, replay_buffer_ON=False):
        self.model_path = model_path
        self.ab_avg_input_path = ab_avg_input_path
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()

        # Start the new standalone worker process
        self.adaptation_process = mp.Process(
            target=adaptation_worker_process, 
            args=(self.input_q, self.output_q, self.model_path, self.ab_avg_input_path, hyperparam_config, replay_buffer_ON)
        )
        self.adaptation_process.start()

    def trigger_finetuning(self, side, input_data, mtr_pos_stream, mid_peak_idx):
        """Sends data to the adaptation worker to start fine-tuning."""
        self.input_q.put((side, input_data, mtr_pos_stream, mid_peak_idx))

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

        if (len(peak_indices) - 1) < 2:
            print(f"Not enough gait cycles detected. {len(peak_indices) - 1} Need at least 2.")
            return
        # else:
            # print(f"\nDetected {len(peak_indices)-1} gait cycles.")
            # if side == 'R':
            #     print(f"R Peak indices: {peak_indices}")
            # else:
            #     print(f"L Peak indices: {peak_indices}")

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