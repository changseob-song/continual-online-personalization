import os, torch, gc, time
import numpy as np
import pandas as pd
import tensorrt as trt
from scipy.signal import butter, filtfilt
from Utils_GPIO import GPIO_control

import pickle

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Fix for NumPy >=2.0 pickles loaded in NumPy <2.0
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module == "numpy.core.multiarray" or module == "numpy.multiarray":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)

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
    upsampled_data = np.zeros((target_length, data.shape[1]))  # (target_length, n_features)
    for i in range(data.shape[1]):
        upsampled_data[:, i] = upsampling(data[:, i], target_length)

    return upsampled_data  # (target_length, n_features)

def get_congruency_rmse_1d(data_1, data_2):
    len_1 = len(data_1)
    len_2 = len(data_2)
    len_rmse = np.abs(len_1 - len_2)
    data_1 = upsampling(data_1, 100)
    data_2 = upsampling(data_2, 100)
    rmse = np.sqrt(np.mean((data_1 - data_2) ** 2))
    return rmse, len_rmse

def get_congruency_rmse_2d(data_1, data_2):
    len_1 = data_1.shape[0]
    len_2 = data_2.shape[0]
    len_rmse = np.abs(len_1 - len_2)
    data_1 = upsampling_2d(data_1, 100)
    data_2 = upsampling_2d(data_2, 100)
    rmse = np.sqrt(np.mean((data_1 - data_2) ** 2))
    return rmse, len_rmse

def causal_filter(data, tau=0.1, dt=0.01, y0=None, return_last=False):
    x = data
    x = np.asarray(x, dtype=float)

    squeeze_1d = False
    if x.ndim == 1:
        x = x[None, :]          # (1, T)
        squeeze_1d = True
    elif x.ndim != 2:
        raise ValueError(f"current shape: {x.shape}. data must be 1D or 2D with shape (C, T).")

    C, T = x.shape

    alpha = dt / float(tau) # alphas is about how much weight we give to the new measurement
    alpha = float(np.clip(alpha, 0.0, 1.0))

    y = np.empty_like(x)
    if y0 is None:
        y[:, 0] = x[:, 0]
    else:
        y0 = np.asarray(y0, dtype=float).reshape(-1)
        y[:, 0] = y0.item() if y0.size == 1 else (
            y0 if y0.size == C else
            (_ for _ in ()).throw(ValueError(f"y0 must be scalar or length {C}"))
        )
    for t in range(1, T):
        y[:, t] = y[:, t-1] + alpha * (x[:, t] - y[:, t-1])

    if return_last:
        out = y[:, -1]
        return out.item() if squeeze_1d else out
    return y.squeeze(0) if squeeze_1d else y

# Lowpass filter class
class lowpass_filter:
    def __init__(self, order=2, cutoff = 4, fs = 100.0):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.nyq = 0.5 * fs
        self.normal_cutoff = cutoff / self.nyq
        self.filter_b, self.filter_a = butter(order, self.normal_cutoff, btype='low', analog=False)

    def apply_lowpass_filter(self, data):
        
        filtered_data = np.zeros_like(data)

        if np.all(data == 0):
            return data

        window_size = data.shape[1]
        # if window_size % 2 == 0:
        #     raise ValueError("Window size should be odd number.")
        half_win = window_size // 2

        for ch in range(data.shape[0]):
            padded_channel = np.pad(data[ch, :], pad_width=(half_win, half_win), mode='reflect')
            filtered_padded = filtfilt(self.filter_b, self.filter_a, padded_channel)
            filtered_data[ch, :] = filtered_padded[half_win:-half_win]

        return filtered_data

def cartesian_to_percentage(cartesian_coords):
    # Ensure input is a numpy array
    if isinstance(cartesian_coords, torch.Tensor):
        cartesian_coords = cartesian_coords.detach().cpu().numpy()
    angle = np.arctan2(cartesian_coords[1], cartesian_coords[0])
    percentage = (angle + 2*np.pi) % (2 * np.pi) / (2 * np.pi) * 100  # Normalize to (0, 100]
    return percentage

# Fast roll function to shift array elements
def fast_roll(arr):
    # For unilateral model
    if len(arr.shape) == 1:
        # 1D arr
        arr[:-1] = arr[1:]
        arr[-1] = 0
    elif len(arr.shape) == 2:
        # 2D arr (e.g. scaled_torque_arr, delayed_torque_arr) (2xframe_length)
        arr[:, :-1] = arr[:, 1:]
        arr[:, -1] = 0
    elif len(arr.shape) == 3:
        # 3D arr (e.g. model_input_arr) (2x14xframe_length)
        arr[:, :, :-1] = arr[:, :, 1:]
        arr[:, :, -1] = 0
    return arr

# TensorRT inference function
def trt_inference(input_data, output_shape, context):
    # Use torch.tensor(...) and torch.empty(...) completely on CUDA:
    d_input = torch.tensor(input_data, dtype=torch.float32, device='cuda')
    d_output = torch.empty(*output_shape, dtype=torch.float32, device='cuda')

    # Prepare bindings
    bindings = [int(d_input.data_ptr()), int(d_output.data_ptr())]

    context.execute_v2(bindings=bindings)

    output = d_output.cpu().numpy()
    return output

# Inference worker function for multiprocessing
def inference_worker(input_q, output_q, trt_engine_path, trt_task_estimator_path,
                     input_mean_path, input_std_path, label_mean_path, label_std_path,
                     input_mean_task_estimator_path, input_std_task_estimator_path, label_mean_task_estimator_path, label_std_task_estimator_path,
                     num_input_features, frame_length, frame_length_task):
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    # Engine for gait phase estimation
    with open(trt_engine_path, 'rb') as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        print("Worker: Failed to deserialize TensorRT engine.")
        return
    context = engine.create_execution_context()

    # Engine for task estimation
    with open(trt_task_estimator_path, 'rb') as f:
        serialized_engine_task = f.read()
    engine_task = runtime.deserialize_cuda_engine(serialized_engine_task)
    if engine_task is None:
        print("Worker: Failed to deserialize TensorRT task estimator engine.")
        return
    context_task = engine_task.create_execution_context()

    dummy_input_data = np.zeros((1, num_input_features, frame_length), dtype=np.float32)
    dummy_input_data_task = np.zeros((1, num_input_features, frame_length_task), dtype=np.float32)
    dummy_output_shape = (80, 100)
    dummy_output_shape_task = (2,)
    for _ in range(10):
        _ = trt_inference(dummy_input_data, dummy_output_shape, context)
    for _ in range(10):
        _ = trt_inference(dummy_input_data_task, dummy_output_shape_task, context_task)
    print("TensorRT engine warmed up.")

    while True:
        try:
            data_in = input_q.get()
            if data_in is None:  # Stop signal
                print("Worker: Stop signal received. Exiting.")
                break

            model_input_arr_l, model_input_arr_r, model_input_arr_task_l, model_input_arr_task_r = data_in

            # Gait phase estimation
            output_shape = (80, 100)  # Assuming scalar output from model
            model_output_l = trt_inference(model_input_arr_l, output_shape, context)
            model_output_r = trt_inference(model_input_arr_r, output_shape, context)

            # Task estimation
            output_shape_task = (2,)  # Assuming scalar output from model
            model_output_task_l = trt_inference(model_input_arr_task_l, output_shape_task, context_task)
            model_output_task_r = trt_inference(model_input_arr_task_r, output_shape_task, context_task) # We assume here that the input frame length is same as gait phase estimator

            output_q.put((model_output_l, model_output_r, model_output_task_l, model_output_task_r))
        except Exception as e:
            print(f"Worker: Error during inference: {e}")
            break
    del context
    del engine
    del runtime
    print("Worker: Exited.")

# Function to save all collected dataif 
def save_data(data_to_save, trial_name, pulse_after_start=0, trial_dur_sec=None):

    # Convert lists to NumPy arrays
    data_np = {k: np.array(v) for k, v in data_to_save.items()}

    # Determine the number of samples to save
    min_len = min(v.shape[0] for v in data_np.values())
    start_idx = int(pulse_after_start * 100)
    end_idx = min(min_len, int((pulse_after_start + trial_dur_sec) * 100)) if trial_dur_sec else min_len

    print(f'Slicing data from index {start_idx} to {end_idx}.')

    # Slice all data arrays
    sliced_data = {k: v[start_idx:end_idx] for k, v in data_np.items()}

    # Define data for motor CSV
    motor_cols = ['timestamp', 'mtr_pos_L', 'mtr_pos_R', 'mtr_vel_L', 'mtr_vel_R', 'fsr_L', 'fsr_R', 'gpio_output']
    save_dataframe(f'{trial_name}_input_motor.csv', sliced_data, motor_cols)

    # Define data for IMU CSV
    imu_df_data = {'timestamp': sliced_data['timestamp']}
    imu_sensors = {'L': 'Thigh_L', 'R': 'Thigh_R'}
    imu_axes = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
    for sensor_code, sensor_name in imu_sensors.items():
        for i, axis_name in enumerate(imu_axes):
            imu_df_data[f'{sensor_name}_{axis_name}'] = sliced_data[f'imu_{sensor_code}'][:, i]

    if 'gpio_output' in sliced_data:
        imu_df_data['gpio_output'] = sliced_data['gpio_output']

    df_imu = pd.DataFrame(imu_df_data)
    df_imu.to_csv(f'{trial_name}_input_imu.csv', index=False)
    print(f'Data saved to {trial_name}_input_imu.csv. Dimensions: {df_imu.shape}')

    # Define data for torque CSV
    torque_cols = ['timestamp', 'gait_phase_L', 'gait_phase_R', 
                   'mtr_cmd_L', 'mtr_cmd_R', 
                   'incline_L', 'speed_L', 'incline_R', 'speed_R', 'gpio_output']
    save_dataframe(f'{trial_name}_output_torque.csv', sliced_data, torque_cols)

# Helper function to create and save DataFrame
def save_dataframe(filename, data_dict, columns):
    df = pd.DataFrame({col: data_dict[col] for col in columns if col in data_dict})
    df.to_csv(filename, index=False)
    print(f'Data saved to {filename}. Dimensions: {df.shape}')

# Function to cleanup CAN resources
def cleanup_can(bus, notifier):
    try:
        notifier.stop()
        bus.shutdown()
        print("CAN resources cleaned up successfully")
    except Exception as e:
        print(f"Error during CAN cleanup: {e}")