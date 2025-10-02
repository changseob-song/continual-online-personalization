# %%
from Hyperparam import hyperparam_config
from Model import TCN
import torch, os, time
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.signal import find_peaks

def adaptation_worker_process(input_q, output_q, model_path, hyperparam_config):
    """
    This worker process handles the fine-tuning of the model.
    All PyTorch and CUDA initializations happen inside this function.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Adaptation Worker: Using device: {device}")

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

    # Main loop to wait for data and fine-tune
    while True:
        try:
            # Wait for data from the main controller
            side, input_data = input_q.get()
            if side is None: # Shutdown signal
                print("Adaptation Worker: Shutdown signal received.")
                break
            
            start_time = time.time()

            # Determine which model and optimizer to use
            model = model_R if side == 'R' else model_L
            optimizer = optimizer_R if side == 'R' else optimizer_L
            
            # Create dataset
            dataset = LoadData(input_data, model_path)

            # If LoadData returns nothing, skip the current update.
            if not dataset.initilized: continue

            train_indices = list(range(len(dataset)))
            subset = Subset(dataset, train_indices)
            train_loader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=3)

            # Training loop
            for input_batch, label_batch in train_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)
                
                optimizer.zero_grad()
                logits = model(input_batch)
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()

            # After training, get the updated weights and send them back
            updated_weights = model.linear.weight.data.clone().cpu().numpy()
            updated_biases = model.linear.bias.data.clone().cpu().numpy()

            output_q.put((side, updated_weights, updated_biases))

            print(f"{side} fine-tuned in {time.time() - start_time:.4f} seconds.")

        except Exception as e:
            print(f"Adaptation worker error: {e}")
            # In case of an error, it's often better to break the loop
            break
    
    print("Adaptation Worker: Exiting.")


class OnlineAdaptator():
    def __init__(self, model_path):
        self.model_path = model_path
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()

        # Start the new standalone worker process
        self.adaptation_process = mp.Process(
            target=adaptation_worker_process, 
            args=(self.input_q, self.output_q, self.model_path, hyperparam_config)
        )
        self.adaptation_process.start()

    def trigger_finetuning(self, side, input_data):
        """Sends data to the adaptation worker to start fine-tuning."""
        self.input_q.put((side, input_data))

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
    def __init__(self, input_data, model_path, gait_cycle_index=None, num_gait_cycles=None):
        self.input = input_data
        self.window_size = hyperparam_config['window_size']
        self.model_path = model_path
        self.initilized = False

        peak_indices, _ = find_peaks(-self.input[:, 6], height= None, distance=15, prominence=10)
        peak_indices = peak_indices.tolist()
        peak_indices.insert(0, 0)  # Add start index
        peak_indices.append(self.input.shape[0])  # Add end index

        print(f"\n{peak_indices}")
        if (len(peak_indices) - 1) < 2:
            print(f"Not enough gait cycles detected. {len(peak_indices) - 1} Need at least 2.")
            return
        else:
            print(f"\nDetected {len(peak_indices)-1} gait cycles.")
            print(f"Peak indices: {peak_indices}")

        def gait_cycle_generator(peak_indices, num_cycles=None):
            gc_angle_list = []
            gc_polar_x_list = []
            gc_polar_y_list = []

            num_cycles = len(peak_indices) - 1
            for i in range(num_cycles):
                start = peak_indices[i]
                end = peak_indices[i + 1]

                for j in range(start, end):
                    gc_polar_angle = (j - start) / (end - start) * 2 * np.pi
                    gc_angle_list.append(gc_polar_angle)
                    gc_polar_x_list.append(np.cos(gc_polar_angle))
                    gc_polar_y_list.append(np.sin(gc_polar_angle))

            return np.column_stack((gc_polar_x_list, gc_polar_y_list))

        # Calculate record time based on the length of the input data
        self.label = gait_cycle_generator(peak_indices)

        # load mean and std for normalization
        input_mean = np.load(os.path.join(os.path.dirname(self.model_path), 'input_mean.npy'))
        input_std = np.load(os.path.join(os.path.dirname(self.model_path), 'input_std.npy'))
        label_mean = np.load(os.path.join(os.path.dirname(self.model_path), 'label_mean.npy'))
        label_std = np.load(os.path.join(os.path.dirname(self.model_path), 'label_std.npy'))

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
        window_input = torch.FloatTensor(windows_input).T           # Shape: (input_size, window_size)
        # print(f"window_input shape: {window_input.shape}")

        # Get the target joint moments at the last time point in the window
        target_label = self.label[idx + self.window_size - 1]       # Shape: (output_size)
            
        window_label = torch.FloatTensor(target_label) # Shape: (output_size), consider putting .T when output_size > 1
        # print(f"window_label shape: {window_label.shape}")
        
        return window_input, window_label