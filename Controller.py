import time, os, atexit, signal, gc, torch
import multiprocessing as mp
import numpy as np

from OnlineAdaptator import OnlineAdaptator
from Utils_Mocap_trigger import Mocap_trigger
from Utils_GPIO import GPIO_control
from Utils_Teleplot import Teleplot
from Utils import lowpass_filter, fast_roll, inference_worker, cleanup_can, save_data, cartesian_to_percentage, NumpyCompatUnpickler, causal_filter
from Exo import Exo
from scipy.signal import find_peaks

class Controller:
    def __init__(self, pt_model_path, trt_engine_path, trt_task_estimator_path, torque_profile_path, ab_avg_input_path,
                 trigger_type, trial_name, pulse_after_start, trial_dur_sec, adjustment_duration, body_mass_kg,
                 adaptation_ON=False, replay_buffer_ON=False):
        self.pt_model_path = pt_model_path
        self.pt_model_linear_path = pt_model_path.replace('.pt', '_linear.pt')
        self.trt_engine_path = trt_engine_path
        self.trt_task_estimator_path = trt_task_estimator_path
        self.ab_avg_input_path = ab_avg_input_path
        self.trigger_type = trigger_type
        self.trial_name = trial_name
        self.body_mass_kg = body_mass_kg
        self.pulse_after_start = pulse_after_start
        self.trial_dur_sec = trial_dur_sec
        self.adjustment_duration = adjustment_duration
        self.adaptation_ON = adaptation_ON
        self.replay_buffer_ON = replay_buffer_ON
        self.last_used_peak_idx_L = 0
        self.last_used_peak_idx_R = 0

        with open(torque_profile_path, "rb") as f:
            self.torque_profile = NumpyCompatUnpickler(f).load()

        # Initialize data structures to save data
        max_samples = int((self.trial_dur_sec + self.pulse_after_start) * 100)
        self.data_to_save = {
            'timestamp': np.zeros(max_samples),
            'mtr_pos_L': np.zeros(max_samples), 'mtr_pos_R': np.zeros(max_samples),
            'mtr_vel_L': np.zeros(max_samples), 'mtr_vel_R': np.zeros(max_samples),
            'imu_L': np.zeros((max_samples, 6)), 'imu_R': np.zeros((max_samples, 6)),
            'mtr_cmd_L': np.zeros(max_samples), 'mtr_cmd_R': np.zeros(max_samples),
            'gait_phase_L': np.zeros(max_samples), 'gait_phase_R': np.zeros(max_samples),
            'incline_L': ['']*max_samples, 'speed_L': ['']*max_samples,
            'incline_R': ['']*max_samples, 'speed_R': ['']*max_samples,
            'gpio_output': np.zeros(max_samples)  # GPIO output state
        }

        # Initialize Teleplot for telemetry data
        self.teleplot = Teleplot()

        # load the normalization values
        base_model_path = os.path.dirname(self.trt_engine_path)
        input_mean_path = os.path.join(base_model_path, 'input_mean.npy')
        input_std_path = os.path.join(base_model_path, 'input_std.npy')
        label_mean_path = os.path.join(base_model_path, 'label_mean.npy')
        label_std_path = os.path.join(base_model_path, 'label_std.npy')

        task_estimator_path = os.path.dirname(self.trt_task_estimator_path)
        input_mean_task_estimator_path = os.path.join(task_estimator_path, 'input_mean.npy')
        input_std_task_estimator_path = os.path.join(task_estimator_path, 'input_std.npy')
        label_mean_task_estimator_path = os.path.join(task_estimator_path, 'label_mean.npy')
        label_std_task_estimator_path = os.path.join(task_estimator_path, 'label_std.npy')

        self.input_mean = np.load(input_mean_path); self.input_std = np.load(input_std_path)
        self.label_mean = np.load(label_mean_path); self.label_std = np.load(label_std_path)
        self.num_input_features = self.input_mean.shape[0]

        self.input_mean_task = np.load(input_mean_task_estimator_path);   self.input_std_task = np.load(input_std_task_estimator_path)
        self.label_mean_task = np.load(label_mean_task_estimator_path);   self.label_std_task = np.load(label_std_task_estimator_path)

        # Initialize the exoskeleton
        if self.trigger_type == "mocap":
            self.mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=10)
            self.mocap_trigger.start_client()
        self.GPIO_control = GPIO_control()
        self.Exo = Exo()
        
        # Initialize lowpass filter
        self.lpf = lowpass_filter()

        # Initialize queues for multiprocessing
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()

        # Start inference_worker process
        self.inference_process = mp.Process(target=inference_worker,
                                    args=(self.input_q, self.output_q, self.trt_engine_path, self.trt_task_estimator_path,
                                            input_mean_path, input_std_path,label_mean_path, label_std_path,
                                            input_mean_task_estimator_path, input_std_task_estimator_path, label_mean_task_estimator_path, label_std_task_estimator_path,
                                            self.num_input_features, self.Exo.frame_length, self.Exo.frame_length_task))
        self.inference_process.start()

        # Extract linear layer weights and biases from the PyTorch model
        state_dict = torch.load(self.pt_model_linear_path, map_location="cpu", weights_only=True)
        self.linear_weights_R = state_dict['weight'].numpy().astype(np.float32)
        self.linear_biases_R = state_dict['bias'].numpy().astype(np.float32)
        self.linear_weights_L = self.linear_weights_R.copy()
        self.linear_biases_L = self.linear_biases_R.copy()

        # Initialize OnlineAdaptator
        self.online_adaptator = OnlineAdaptator(self.pt_model_path, self.ab_avg_input_path, self.adaptation_ON, self.replay_buffer_ON)

    def run_loop(self, Exo_ON=False):

        # Setting for the exiting process
        atexit.register(lambda: (cleanup_can(self.Exo.bus, self.Exo.notifier), self.GPIO_control.safe_gpio_cleanup()))
        signal.signal(signal.SIGINT, self.exit_signal_handler)

        # Rolling array initialization of model output array (2xframe_length)
        model_input_arr = np.zeros((2, self.num_input_features, self.Exo.frame_length), dtype=np.float32)
        # Predefine input stream data size for online adaptation (6 seconds buffer)
        input_stream_data = np.zeros((2, self.num_input_features, 6 * self.Exo.control_freq_Hz), dtype=np.float32)  # 100 frames of input data
        motor_cmd_array = np.zeros((2, self.Exo.frame_length), dtype=np.float32)  # for torque command filtering
        incline_pred_history = np.zeros((2, self.Exo.frame_length_task), dtype=np.float32)
        speed_pred_history = np.zeros((2, self.Exo.frame_length_task), dtype=np.float32)

        current_pos_L, current_vel_L = 0.0, 0.0
        current_pos_R, current_vel_R = 0.0, 0.0
        gait_phase_R_prev = 0.0
        gait_phase_L_prev = 0.0

        local_l_data = np.zeros(6); local_r_data = np.zeros(6)

        right_data = np.zeros(self.num_input_features); left_data = np.zeros(self.num_input_features)

        last_model_output_r = np.zeros((80, 100), dtype=np.float32); last_model_output_l = np.zeros((80, 100), dtype=np.float32)
        last_model_output_r_task = np.zeros((2,), dtype=np.float32); last_model_output_l_task = np.zeros((2,), dtype=np.float32)
        model_output_r_val = last_model_output_r; model_output_l_val = last_model_output_l
        model_output_r_task = last_model_output_r_task; model_output_l_task = last_model_output_l_task

        incline_keys = {-10: "RD_10deg", -5: "RD_5deg", 0: "LG", 5: "RA_5deg", 10: "RA_10deg"}
        speed_keys = {0.2: "0p2mps", 0.4: "0p4mps", 0.6: "0p6mps", 0.8: "0p8mps", 1.0: "1p0mps", 1.2: "1p2mps", 1.4: "1p4mps"}
        incline_thresholds = [-7.5, -2.5, 2.5, 7.5]
        incline_values = [-10, -5, 0, 5, 10]
        speed_thresholds = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
        speed_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

        update_time_R, update_time_L = 0.0, 0.0
        avg_loss_R, avg_loss_L = 0.0, 0.0

        # Create local references to data arrays for faster access
        log_timestamp = self.data_to_save['timestamp']
        log_mtr_pos_L, log_mtr_pos_R = self.data_to_save['mtr_pos_L'], self.data_to_save['mtr_pos_R']
        log_mtr_vel_L, log_mtr_vel_R = self.data_to_save['mtr_vel_L'], self.data_to_save['mtr_vel_R']
        log_imu_L, log_imu_R = self.data_to_save['imu_L'], self.data_to_save['imu_R']
        log_mtr_cmd_L, log_mtr_cmd_R = self.data_to_save['mtr_cmd_L'], self.data_to_save['mtr_cmd_R']
        log_gait_phase_L, log_gait_phase_R = self.data_to_save['gait_phase_L'], self.data_to_save['gait_phase_R']

        log_incline_L, log_speed_L = self.data_to_save['incline_L'], self.data_to_save['speed_L']
        log_incline_R, log_speed_R = self.data_to_save['incline_R'], self.data_to_save['speed_R']
        log_gpio_output = self.data_to_save['gpio_output']

        current_incline_L = 'LG'; current_speed_L = '0p2mps'  # Default task settings
        current_incline_L_val = 0; current_speed_L_val = 0
        prev_incline_L = current_incline_L; prev_speed_L = current_speed_L
        current_incline_R = 'LG'; current_speed_R = '0p2mps'  # Default task settings
        current_incline_R_val = 0; current_speed_R_val = 0
        prev_incline_R = current_incline_R; prev_speed_R = current_speed_R

        # Start recording time
        first_pulse_sent = False
        first_pulse_end_time = None
        second_pulse_sent = False
        second_pulse_end_time = None
        loop_index = 1

        # Wait for the trigger to start the trial
        if self.trigger_type == "mocap":
            print("Wait for the tensorrt to warm up...\n")
            self.mocap_trigger.wait_for_trigger()
            print("Mocap trigger received - starting data logging")
            
        elif self.trigger_type == "typing":
            input_trigger = input("Wait for the tensorrt to warm up...\n")
            if input_trigger == "":
                print("Trial started")

        start_time = time.time()

        # Main control loop
        while True:

            log_incline_L[loop_index] = current_incline_L; log_speed_L[loop_index] = current_speed_L
            log_incline_R[loop_index] = current_incline_R; log_speed_R[loop_index] = current_speed_R

            # 1. Read the motor encoder values
            current_pos_L, current_vel_L = self.Exo.update_readings(self.Exo.CAN_id_L)
            current_pos_R, current_vel_R = self.Exo.update_readings(self.Exo.CAN_id_R)

            current_pos_R *= -1; current_vel_R *= -1 # mirror the right side values (because of the motor mounting direction)

            log_mtr_pos_L[loop_index] = current_pos_L; log_mtr_pos_R[loop_index] = current_pos_R
            log_mtr_vel_L[loop_index] = current_vel_L; log_mtr_vel_R[loop_index] = current_vel_R

            # 2. Read the IMU values
            imu_dict = self.Exo.imus.read_IMUs()

            local_l_data = imu_dict["IMU_THIGH_LEFT"]; local_r_data = imu_dict["IMU_THIGH_RIGHT"]
            log_imu_L[loop_index, :] = local_l_data; log_imu_R[loop_index, :] = local_r_data

            # 3. Mirror the left data to the right side
            l_data_reflected = local_l_data.copy()
            l_data_reflected[1] *= -1; l_data_reflected[3] *= -1; l_data_reflected[5] *= -1
            
            # 4. Prepare the model input data
            right_data[:6] = local_r_data; #right_data[6] = current_pos_R
            left_data[:6] = l_data_reflected; #left_data[6] = current_pos_L

            right_data_norm = (right_data - self.input_mean) / self.input_std
            left_data_norm = (left_data - self.input_mean) / self.input_std

            model_input_arr = fast_roll(model_input_arr)
            model_input_arr[0, :, -1] = right_data_norm; model_input_arr[1, :, -1] = left_data_norm

            # 4.1 Prepare the input data for online adaptation
            if first_pulse_sent:

                input_stream_data = fast_roll(input_stream_data)
                input_stream_data[0, :, -1] = right_data; input_stream_data[1, :, -1] = left_data

                # --- REVISED ADAPTATION TRIGGER LOGIC ---
                update_freq_gc = 2 # Number of gait cycles for each adaptation update
                
                # Optimized peak detection on recent data
                search_window = 600 # Search in the last 6 seconds
                search_start_idx = max(0, loop_index - search_window)
                recent_pos_L = self.data_to_save['mtr_pos_L'][search_start_idx:loop_index]
                recent_pos_R = self.data_to_save['mtr_pos_R'][search_start_idx:loop_index]

                peak_indices_L, _ = find_peaks(-recent_pos_L, height=None, distance=40, prominence=15)
                peak_indices_L += search_start_idx # This makes the indices back to absolute timeframe
                peak_indices_R, _ = find_peaks(-recent_pos_R, height=None, distance=40, prominence=15)
                peak_indices_R += search_start_idx
                
                buffer_start_abs = loop_index - len(input_stream_data[0, 0, :])

                # Check if there are enough new peaks for an update (2 gait cycles = 2 new peaks after the start)
                if self.last_used_peak_idx_R not in peak_indices_R:
                    if len(peak_indices_R) > update_freq_gc:
                        # Get the absolute start and end indices for the data slice
                        start_idx_abs = peak_indices_R[0]; end_idx_abs = peak_indices_R[-1]
                        print('\nR', start_idx_abs, peak_indices_R[-2], end_idx_abs)
                        mid_peak_idx_rel = peak_indices_R[1:-1] - start_idx_abs # This is relative about start_idx_abs

                        # Slice the data from the input stream buffer
                        start_idx_rel = start_idx_abs - buffer_start_abs; end_idx_rel = end_idx_abs - buffer_start_abs
                        input_stream_data_R = input_stream_data[0, :, start_idx_rel:end_idx_rel]
                        mtr_pos_stream_R = log_mtr_pos_R[start_idx_abs:end_idx_abs]

                        self.online_adaptator.trigger_finetuning('R', input_stream_data_R.T.copy(), mtr_pos_stream_R, mid_peak_idx_rel)

                        # Update the last used peak to the end of the current window
                        self.last_used_peak_idx_R = end_idx_abs

                # Check if there are enough new peaks for an update (2 gait cycles = 2 new peaks after the start)
                if self.last_used_peak_idx_L not in peak_indices_L:
                    if len(peak_indices_L) > update_freq_gc:
                        # Get the absolute start and end indices for the data slice
                        start_idx_abs = peak_indices_L[0]; end_idx_abs = peak_indices_L[-1]
                        print('\nL', start_idx_abs, peak_indices_L[-2], end_idx_abs)
                        mid_peak_idx_rel = peak_indices_L[1:-1] - start_idx_abs # This is relative about start_idx_abs

                        # Slice the data from the input stream buffer
                        start_idx_rel = start_idx_abs - buffer_start_abs; end_idx_rel = end_idx_abs - buffer_start_abs
                        input_stream_data_L = input_stream_data[1, :, start_idx_rel:end_idx_rel]
                        mtr_pos_stream_L = log_mtr_pos_L[start_idx_abs:end_idx_abs]

                        self.online_adaptator.trigger_finetuning('L', input_stream_data_L.T.copy(), mtr_pos_stream_L, mid_peak_idx_rel)

                        # Update the last used peak to the end of the current window
                        self.last_used_peak_idx_L = end_idx_abs

            # Check for new weights from the adaptation worker
            updated_params = self.online_adaptator.get_updated_weights()
            if updated_params:
                side, current_incline, current_speed, avg_loss, update_time, weights, biases = updated_params
                if side == 'R':
                    self.linear_weights_R = weights;    self.linear_biases_R = biases
                    # current_incline_R = current_incline; current_speed_R = current_speed
                    avg_loss_R = avg_loss;  update_time_R = update_time
                elif side == 'L':
                    self.linear_weights_L = weights;    self.linear_biases_L = biases
                    # current_incline_L = current_incline; current_speed_L = current_speed
                    avg_loss_L = avg_loss;  update_time_L = update_time

            # 5. TensorRT inference & Apply linear layer weights and biases
            self.input_q.put((model_input_arr[0, :, :].copy(), model_input_arr[1, :, :].copy()))
            try:
                model_output_r_val, model_output_l_val, model_output_r_task, model_output_l_task = self.output_q.get_nowait()
                last_model_output_r, last_model_output_l = model_output_r_val, model_output_l_val
                last_model_output_r_task, last_model_output_l_task = model_output_r_task, model_output_l_task
            except mp.queues.Empty:
                model_output_r_val, model_output_l_val = last_model_output_r, last_model_output_l
                model_output_r_task, model_output_l_task = last_model_output_r_task, last_model_output_l_task
            
            # Apply linear layer weights and biases
            model_output_r_val = np.dot(model_output_r_val.flatten(), self.linear_weights_R.T)
            model_output_r_val += self.linear_biases_R

            model_output_l_val = np.dot(model_output_l_val.flatten(), self.linear_weights_L.T)
            model_output_l_val += self.linear_biases_L

            # 6. Calculate the gait phase
            model_output_r_denorm = model_output_r_val * self.label_std + self.label_mean
            model_output_l_denorm = model_output_l_val * self.label_std + self.label_mean

            gait_phase_R = cartesian_to_percentage(model_output_r_denorm)
            gait_phase_L = cartesian_to_percentage(model_output_l_denorm)

            # Store previous gait phase if decreasing
            if (gait_phase_R < gait_phase_R_prev) and (gait_phase_R_prev < 95): gait_phase_R = gait_phase_R_prev
            else: gait_phase_R_prev = gait_phase_R
            if (gait_phase_L < gait_phase_L_prev) and (gait_phase_L_prev < 95): gait_phase_L = gait_phase_L_prev
            else: gait_phase_L_prev = gait_phase_L

            delayed_gait_phase_R = (gait_phase_R - self.Exo.delay_factor) % 100
            delayed_gait_phase_L = (gait_phase_L - self.Exo.delay_factor) % 100

            # if loop_index / self.Exo.control_freq_Hz >= self.pulse_after_start:
            #     gradual_torque_scale = min(1.0, ((loop_index / self.Exo.control_freq_Hz) - self.pulse_after_start) / self.adjustment_duration)
            # else:
            #     gradual_torque_scale = 0.0

            gradual_torque_scale_L = 1 # np.max((1 - avg_loss_L), 0)
            gradual_torque_scale_R = 1 # np.max((1 - avg_loss_R), 0)

            # 6.1 Get the task estimation outputs
            model_output_r_task_denorm = model_output_r_task * self.label_std_task + self.label_mean_task
            model_output_l_task_denorm = model_output_l_task * self.label_std_task + self.label_mean_task

            incline_pred_R = model_output_r_task_denorm[0]; incline_pred_L = model_output_l_task_denorm[0]
            speed_pred_R = model_output_r_task_denorm[1]; speed_pred_L = model_output_l_task_denorm[1]

            incline_pred_history = fast_roll(incline_pred_history); speed_pred_history = fast_roll(speed_pred_history)
            incline_pred_history[0, -1] = incline_pred_R; incline_pred_history[1, -1] = incline_pred_L
            speed_pred_history[0, -1] = speed_pred_R; speed_pred_history[1, -1] = speed_pred_L

            # 7. Send the torque command to the motors
            if delayed_gait_phase_L < 5:
                inc_pred_mean = np.mean(incline_pred_history[1, :]); spd_pred_mean = np.mean(speed_pred_history[1, :])
                # Discretize incline prediction
                current_incline_L_val = incline_values[np.searchsorted(incline_thresholds, inc_pred_mean)]
                current_speed_L_val = speed_values[np.searchsorted(speed_thresholds, spd_pred_mean)]

                current_incline_L = incline_keys.get(current_incline_L_val, prev_incline_L)
                current_speed_L = speed_keys.get(current_speed_L_val, prev_speed_L)
                if current_incline_L != 'LG' and current_speed_L not in ['0p4mps', '0p6mps', '0p8mps', '1p0mps']:
                    current_incline_L = prev_incline_L; current_speed_L = prev_speed_L
                elif current_incline_L == 'RD_10deg' and current_speed_L in ['0p4mps', '0p6mps', '1p2mps']:
                    current_incline_L = prev_incline_L; current_speed_L = prev_speed_L

                motor_cmd_val_L = self.torque_profile[current_incline_L][current_speed_L][int(delayed_gait_phase_L)] * self.body_mass_kg * self.Exo.scale_factor * gradual_torque_scale_L
                prev_incline_L = current_incline_L; prev_speed_L = current_speed_L
            else:
                motor_cmd_val_L = self.torque_profile[prev_incline_L][prev_speed_L][int(delayed_gait_phase_L)] * self.body_mass_kg * self.Exo.scale_factor * gradual_torque_scale_L
            
            if delayed_gait_phase_R < 5:
                inc_pred_mean = np.mean(incline_pred_history[0, :]); spd_pred_mean = np.mean(speed_pred_history[0, :])
                # Discretize incline prediction
                current_incline_R_val = incline_values[np.searchsorted(incline_thresholds, inc_pred_mean)]
                current_speed_R_val = speed_values[np.searchsorted(speed_thresholds, spd_pred_mean)]

                current_incline_R = incline_keys.get(current_incline_R_val, prev_incline_R)
                current_speed_R = speed_keys.get(current_speed_R_val, prev_speed_R)
                if current_incline_R != 'LG' and current_speed_R not in ['0p4mps', '0p6mps', '0p8mps', '1p0mps']:
                    current_incline_R = prev_incline_R; current_speed_R = prev_speed_R
                elif current_incline_R == 'RD_10deg' and current_speed_R in ['0p4mps', '0p6mps', '1p2mps']:
                    current_incline_R = prev_incline_R; current_speed_R = prev_speed_R

                motor_cmd_val_R = self.torque_profile[current_incline_R][current_speed_R][int(delayed_gait_phase_R)] * self.body_mass_kg * self.Exo.scale_factor * gradual_torque_scale_R
                prev_incline_R = current_incline_R; prev_speed_R = current_speed_R
            else:
                motor_cmd_val_R = self.torque_profile[prev_incline_R][prev_speed_R][int(delayed_gait_phase_R)] * self.body_mass_kg * self.Exo.scale_factor * gradual_torque_scale_R

            motor_cmd_array = fast_roll(motor_cmd_array)
            motor_cmd_array[:, -1] = [motor_cmd_val_R, motor_cmd_val_L]    

            # 8. Filter the torque command
            # motor_cmd_val_L = causal_filter(motor_cmd_array[1, :], tau=0.05)[-1]
            # motor_cmd_val_R = causal_filter(motor_cmd_array[0, :], tau=0.05)[-1]

            if Exo_ON == False: motor_cmd_val_L, motor_cmd_val_R = 0.0, 0.0 # use this for Exo off condition

            if motor_cmd_val_L > 8:    motor_cmd_val_L = 8
            elif motor_cmd_val_L < -8: motor_cmd_val_L = -8
            if motor_cmd_val_R > 8:    motor_cmd_val_R = 8
            elif motor_cmd_val_R < -8: motor_cmd_val_R = -8

            self.Exo.mtr_comms.set_torque(self.Exo.CAN_id_L, motor_cmd_val_L) 
            self.Exo.mtr_comms.set_torque(self.Exo.CAN_id_R, -motor_cmd_val_R) # Negative sign because the motor is mounted in reverse direction

            # 8. Stack the data (that will be saved after the trial)
            log_mtr_cmd_L[loop_index] = motor_cmd_val_L
            log_mtr_cmd_R[loop_index] = motor_cmd_val_R

            log_gait_phase_L[loop_index] = gait_phase_L
            log_gait_phase_R[loop_index] = gait_phase_R

            # GPIO pulse logic starting from here
            current_time = time.time() - start_time
            
            # First pulse
            if current_time >= (self.pulse_after_start) and not first_pulse_sent:
                self.GPIO_control.send_gpio_pulse_start()
                first_pulse_sent = True
                first_pulse_end_time = current_time + 0.2  # 200ms pulse duration
                print("First pulse sent")
            # First pulse end
            if first_pulse_sent and first_pulse_end_time and current_time >= first_pulse_end_time:
                self.GPIO_control.send_gpio_pulse_end()
                first_pulse_end_time = None
                print("First pulse ended")
            
            # Second pulse
            if current_time >= (self.pulse_after_start + self.trial_dur_sec) and not second_pulse_sent:
                self.GPIO_control.send_gpio_pulse_start()
                second_pulse_sent = True
                second_pulse_end_time = current_time + 0.2  # 200ms pulse duration
                print("Second pulse sent")
            
            # Second pulse end
            if second_pulse_sent and second_pulse_end_time and current_time >= second_pulse_end_time:
                self.GPIO_control.send_gpio_pulse_end()
                second_pulse_end_time = None
                print("Second pulse ended")

                break # Exit the loop after the second pulse ends

            # GPIO output logging
            log_gpio_output[loop_index] = self.GPIO_control.get_gpio_output_state()

            # 9. Loop time
            loop_time_exceeded = (time.time() - start_time) - (loop_index / self.Exo.control_freq_Hz)

            # 10. Send telemetry data
            telemetry_data = {
                "pos_L": current_pos_L,
                "pos_R": current_pos_R,
                "gyroY_L": local_l_data[4],
                "gyroY_R": local_r_data[4],
                "gait_phase_L": gait_phase_L,
                "gait_phase_R": gait_phase_R,
                "incline_L_cont": incline_pred_L,
                "incline_R_cont": incline_pred_R,
                "speed_L_cont": speed_pred_L,
                "speed_R_cont": speed_pred_R,
                "incline_L_disc": current_incline_L_val,
                "incline_R_disc": current_incline_R_val,
                "speed_L_disc": current_speed_L_val,
                "speed_R_disc": current_speed_R_val,
                "cmd_L": motor_cmd_val_L,
                "cmd_R": motor_cmd_val_R,
                "update_time_L": update_time_L,
                "update_time_R": update_time_R,
                "loop_time_exceeded": loop_time_exceeded,
            }
            self.teleplot.sendBatchTelemetry(telemetry_data)

            # 11. Wait for the time to reach the next clock cycle
            if (time.time() - start_time) < (loop_index / self.Exo.control_freq_Hz):
                while (time.time() - start_time) < (loop_index / self.Exo.control_freq_Hz):
                    pass
            log_timestamp[loop_index] = time.time() - start_time
            loop_index += 1

    # Signal handler for graceful exit
    def exit_signal_handler(self, sig, frame):
        print("Ctrl + C pressed, shutting down...")

        # Apply zero torque to the motors
        self.Exo.mtr_comms.set_torque(self.Exo.CAN_id_L, 0)
        self.Exo.mtr_comms.set_torque(self.Exo.CAN_id_R, 0)

        save_data(self.data_to_save, self.trial_name, self.pulse_after_start, self.trial_dur_sec)
        cleanup_can(self.Exo.bus, self.Exo.notifier)
        self.GPIO_control.safe_gpio_cleanup()

        gc.collect()
        torch.cuda.empty_cache()

        print("Exiting program")
        os._exit(0)