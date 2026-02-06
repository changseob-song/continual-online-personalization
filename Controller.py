import time, os, atexit, signal, gc, torch
import multiprocessing as mp
import numpy as np
import pandas as pd

from OnlineAdaptator import OnlineAdaptator
from Utils_Mocap_Datastream import Mocap_trigger
from Utils_GPIO import GPIO_control
from Utils_Teleplot import Teleplot
from Utils import lowpass_filter, fast_roll, gait_phase_inference_worker, cleanup_can, save_data, save_weights_biases, cartesian_to_percentage, NumpyCompatUnpickler, causal_filter
from Exo import Exo
from scipy.signal import find_peaks

class Controller:
    def __init__(self, pt_model_path, trt_engine_path, linear_layer_path, torque_profile_path, pca_model_path,
                 trigger_type, trial_name, course_num, incline, pulse_after_start, trial_dur_sec, adjustment_duration, body_mass_kg,
                 adaptation_ON=False, replay_buffer_ON=False):
        self.pt_model_path = pt_model_path
        self.pt_model_linear_path = pt_model_path.replace('.pt', '_linear.pt')
        self.linear_layer_path = linear_layer_path
        self.trt_engine_path = trt_engine_path
        self.pca_model_path = pca_model_path
        self.trigger_type = trigger_type
        self.trial_name = trial_name
        self.course_num = course_num
        self.incline = incline
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

        # Initialize Teleplot for telemetry data
        self.teleplot = Teleplot()

        # load the normalization values
        base_model_path = os.path.dirname(self.trt_engine_path)
        input_mean_path = os.path.join(base_model_path, 'input_mean.npy')
        input_std_path = os.path.join(base_model_path, 'input_std.npy')
        label_mean_path = os.path.join(base_model_path, 'label_mean.npy')
        label_std_path = os.path.join(base_model_path, 'label_std.npy')

        self.input_mean = np.load(input_mean_path); self.input_std = np.load(input_std_path)
        self.label_mean = np.load(label_mean_path); self.label_std = np.load(label_std_path)

        self.num_input_features = self.input_mean.shape[0]

        # Initialize the exoskeleton
        if self.trigger_type == "mocap":
            self.mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=11)
            self.mocap_trigger.start_client()
        self.GPIO_control = GPIO_control()
        self.Exo = Exo()
        
        # Initialize lowpass filter
        self.lpf = lowpass_filter()

        # Initialize queues for multiprocessing
        self.gait_phase_input_q = mp.Queue()
        self.gait_phase_output_q = mp.Queue()

        # Start gait_phase_inference_worker process
        self.gait_phase_inference_process = mp.Process(target=gait_phase_inference_worker,
                                            args=(self.gait_phase_input_q, self.gait_phase_output_q, self.trt_engine_path,
                                                  self.num_input_features, self.Exo.frame_length))
        self.gait_phase_inference_process.start()

        # For the first course, load the pre-trained one
        if self.course_num == 1:
            # Extract linear layer weights and biases from the PyTorch model
            state_dict = torch.load(self.pt_model_linear_path, map_location="cpu", weights_only=True)
            self.linear_weights_R = state_dict['weight'].numpy().astype(np.float32)
            self.linear_biases_R = state_dict['bias'].numpy().astype(np.float32)
            self.linear_weights_L = self.linear_weights_R.copy()
            self.linear_biases_L = self.linear_biases_R.copy()
        # For the subsequent courses, load from the saved weight& bias file
        else:
            with open(self.linear_layer_path, "rb") as f:
                linear_params = NumpyCompatUnpickler(f).load()
            self.linear_weights_R = linear_params['weights_R'].astype(np.float32)
            self.linear_biases_R = linear_params['biases_R'].astype(np.float32)
            self.linear_weights_L = linear_params['weights_L'].astype(np.float32)
            self.linear_biases_L = linear_params['biases_L'].astype(np.float32)

        # Initialize OnlineAdaptator
        self.online_adaptator = OnlineAdaptator(self.pt_model_path, self.pca_model_path, self.adaptation_ON, self.replay_buffer_ON)

        self.incline_values = [-10, -5, 0, 5, 10]
        self.incline_keys = {'RD_10': -10, 'RD_5': -5, 'LG': 0, 'RA_5': 5, 'RA_10': 10}

    def detect_heel_strike(self, GRF_data, threshold, min_interval):
        # Create binary GRF signal based on threshold
        bi_GRF = np.where(GRF_data < threshold, 0, 1)
        diff_GRF = np.diff(bi_GRF)
        # Find indices where the signal goes from 0 to 1
        heelstrike_index = np.where(diff_GRF == 1)[0] + 1
        if heelstrike_index.size == 0:
            return np.array([], dtype=int)
        valid_heelstrikes = heelstrike_index[np.insert(np.diff(heelstrike_index) >= min_interval, 0, True)]
        return valid_heelstrikes

    def run_loop(self, Exo_ON=False):

        # Setting for the exiting process
        atexit.register(lambda: (cleanup_can(self.Exo.bus, self.Exo.notifier), self.GPIO_control.safe_gpio_cleanup()))
        signal.signal(signal.SIGINT, self.exit_signal_handler)

        # Rolling array initialization of model output array (2xframe_length)
        model_input_arr = np.zeros((2, self.num_input_features, self.Exo.frame_length), dtype=np.float32)
        # Predefine input stream data size for online adaptation (6 seconds buffer)
        input_stream_data = np.zeros((2, self.num_input_features, 6 * self.Exo.control_freq_Hz), dtype=np.float32)  # 100 frames of input data
        motor_cmd_array = np.zeros((2, self.Exo.frame_length), dtype=np.float32)  # for torque command filtering

        mtr_pos_L, mtr_pos_R = 0.0, 0.0
        mtr_vel_L, mtr_vel_R = 0.0, 0.0
        imu_L, imu_R = np.zeros(6), np.zeros(6)
        GRF_L, GRF_R = 0.0, 0.0

        left_data, right_data = np.zeros(self.num_input_features), np.zeros(self.num_input_features)
        gait_phase_L_prev, gait_phase_R_prev = 0.0, 0.0

        last_model_output_r = np.zeros((80, 100), dtype=np.float32); last_model_output_l = np.zeros((80, 100), dtype=np.float32)
        model_output_r_val = last_model_output_r; model_output_l_val = last_model_output_l

        update_latency_R, update_latency_L = 0.0, 0.0
        gp_loop_index = 0
        start_idx_L, start_idx_R = -1, -1

        current_incline = self.incline_keys[self.incline];  # Default task settings
        prev_incline = current_incline

        # Initialize data structures to save data
        max_samples = int((self.trial_dur_sec + self.pulse_after_start + 5) * 100) # Extra 5 sec for slowing down
        self.data_to_save = {
            'timestamp': np.zeros(max_samples),
            'mtr_pos_L': np.zeros(max_samples), 'mtr_pos_R': np.zeros(max_samples),
            'mtr_vel_L': np.zeros(max_samples), 'mtr_vel_R': np.zeros(max_samples),
            'imu_L': np.zeros((max_samples, 6)), 'imu_R': np.zeros((max_samples, 6)),
            'GRF_L': np.zeros(max_samples), 'GRF_R': np.zeros(max_samples),
            'mtr_cmd_L': np.zeros(max_samples), 'mtr_cmd_R': np.zeros(max_samples),
            'gait_phase_L': np.zeros(max_samples), 'gait_phase_R': np.zeros(max_samples),
            'incline': ['']*max_samples, 'speed': ['']*max_samples,
            'input_reduced_R': ['']*max_samples, 'input_reduced_L': ['']*max_samples,
            'grid_key_R': ['']*max_samples, 'grid_key_L': ['']*max_samples,
            'gpio_output': np.zeros(max_samples)  # GPIO output state
        }

        # Create local references to data arrays for faster access
        log_timestamp = self.data_to_save['timestamp']
        log_mtr_pos_L, log_mtr_pos_R = self.data_to_save['mtr_pos_L'], self.data_to_save['mtr_pos_R']
        log_mtr_vel_L, log_mtr_vel_R = self.data_to_save['mtr_vel_L'], self.data_to_save['mtr_vel_R']
        log_imu_L, log_imu_R = self.data_to_save['imu_L'], self.data_to_save['imu_R']
        log_GRF_L, log_GRF_R = self.data_to_save['GRF_L'], self.data_to_save['GRF_R']
        log_mtr_cmd_L, log_mtr_cmd_R = self.data_to_save['mtr_cmd_L'], self.data_to_save['mtr_cmd_R']
        log_gait_phase_L, log_gait_phase_R = self.data_to_save['gait_phase_L'], self.data_to_save['gait_phase_R']
        log_incline = self.data_to_save['incline']; log_speed = self.data_to_save['speed']

        log_input_reduced_L = self.data_to_save['input_reduced_L']; log_input_reduced_R = self.data_to_save['input_reduced_R']
        log_grid_key_L = self.data_to_save['grid_key_L']; log_grid_key_R = self.data_to_save['grid_key_R']
        log_gpio_output = self.data_to_save['gpio_output']

        # Start recording time
        first_pulse_sent = False
        first_pulse_end_time = None
        second_pulse_sent = False
        second_pulse_end_time = None
        loop_index = 0

        # Wait for the trigger to start the trial
        if self.trigger_type == "mocap":
            print("Wait for the tensorrt to warm up...\n")
            self.mocap_trigger.start_client()
            self.mocap_trigger.stream_start()
            self.mocap_trigger.wait_for_start_logging()
            print("Mocap trigger received - starting data logging")
            
        elif self.trigger_type == "typing":
            input_trigger = input("Wait for the tensorrt to warm up...\n")
            if input_trigger == "":
                print("Trial started")

        start_time = time.time()

        # Main control loop
        while True:

            # 1. Read the motor encoder values
            mtr_pos_L, mtr_vel_L = self.Exo.update_readings(self.Exo.CAN_id_L)
            mtr_pos_R, mtr_vel_R = self.Exo.update_readings(self.Exo.CAN_id_R)
            mtr_pos_R *= -1; mtr_vel_R *= -1 # mirror the right side values (because of the motor mounting direction)
            log_mtr_pos_L[loop_index] = mtr_pos_L; log_mtr_pos_R[loop_index] = mtr_pos_R
            log_mtr_vel_L[loop_index] = mtr_vel_L; log_mtr_vel_R[loop_index] = mtr_vel_R

            # 2. Read the IMU values
            imu_dict = self.Exo.imus.read_IMUs()
            imu_L, imu_R = imu_dict["IMU_THIGH_LEFT"], imu_dict["IMU_THIGH_RIGHT"]
            log_imu_L[loop_index, :], log_imu_R[loop_index, :] = imu_L, imu_R

            # 2.1 Read the GRF values
            GRF_L, GRF_R, current_speed = self.mocap_trigger.get_GRF()
            # GRF_L, GRF_R, current_speed = 0, 0, 0.6
            log_GRF_L[loop_index] = GRF_L; log_GRF_R[loop_index] = GRF_R
            log_incline[loop_index] = current_incline; log_speed[loop_index] = current_speed

            # 3. Mirror the left data to the right side (Unilateral model input)
            imu_L_reflected, imu_R_reflected = imu_L.copy(), imu_R.copy()
            imu_L_reflected[1] *= -1; imu_L_reflected[3] *= -1; imu_L_reflected[5] *= -1
            imu_R_reflected[1] *= -1; imu_R_reflected[3] *= -1; imu_R_reflected[5] *= -1

            # 4. Prepare the model input data
            left_data, right_data = np.array([mtr_pos_L, mtr_vel_L, mtr_pos_R, mtr_vel_R]), np.array([mtr_pos_R, mtr_vel_R, mtr_pos_L, mtr_vel_L])
            # left_data, right_data = np.concatenate([imu_L_reflected, imu_R_reflected]), np.concatenate([imu_R, imu_L])

            left_data_norm, right_data_norm = (left_data - self.input_mean) / self.input_std, (right_data - self.input_mean) / self.input_std

            model_input_arr = fast_roll(model_input_arr)
            model_input_arr[0, :, -1], model_input_arr[1, :, -1] = left_data_norm, right_data_norm
            
            # 4.1 Prepare the input data for online adaptation
            if first_pulse_sent:

                input_stream_data = fast_roll(input_stream_data)
                input_stream_data[0, :, -1], input_stream_data[1, :, -1] = left_data, right_data

                # --- REVISED ADAPTATION TRIGGER LOGIC ---
                update_freq_gc = 2 # Number of gait cycles for each adaptation update
                num_skipped_cycles = 1 # Number of initial cycles to skip after starting adaptation

                # Optimized peak detection on recent data
                search_window = 700 # Search in the last 5 seconds
                search_start_idx = max(0, loop_index - search_window)
                recent_GRF_L = self.data_to_save['GRF_L'][search_start_idx:loop_index]
                recent_GRF_R = self.data_to_save['GRF_R'][search_start_idx:loop_index]

                heelstrike_indices_L = self.detect_heel_strike(recent_GRF_L, 0.1, min_interval=50)
                heelstrike_indices_L += (search_start_idx)  # Convert to absolute indices
                heelstrike_indices_R = self.detect_heel_strike(recent_GRF_R, 0.1, min_interval=50)
                heelstrike_indices_R += (search_start_idx)  # Convert to absolute indices

                buffer_start_abs = loop_index - len(input_stream_data[0, 0, :])

                # Check if there are enough new heel strikes for an update (2 gait cycles = 2 new heel strikes after the start)
                if (self.last_used_peak_idx_L not in heelstrike_indices_L):
                    if len(heelstrike_indices_L) > update_freq_gc + num_skipped_cycles:
                        # Get the absolute start and end indices for the data slice
                        start_idx_abs = heelstrike_indices_L[-3]; end_idx_abs = heelstrike_indices_L[-1]
                        print('\nL', start_idx_abs, heelstrike_indices_L[-2:-1], end_idx_abs)
                        mid_peak_idx_rel = heelstrike_indices_L[-2:-1] - start_idx_abs # This is relative about start_idx_abs

                        # Slice the data from the input stream buffer
                        start_idx_rel = start_idx_abs - buffer_start_abs; end_idx_rel = end_idx_abs - buffer_start_abs
                        input_stream_data_L = input_stream_data[0, :, start_idx_rel:end_idx_rel]

                        self.online_adaptator.trigger_finetuning('L', current_incline, current_speed, input_stream_data_L.T.copy(), mid_peak_idx_rel, loop_index)

                        # Update the last used peak to the end of the current window
                        self.last_used_peak_idx_L = end_idx_abs

                # Check if there are enough new heel strikes for an update (2 gait cycles = 2 new heel strikes after the start)
                if (self.last_used_peak_idx_R not in heelstrike_indices_R):
                    if len(heelstrike_indices_R) > update_freq_gc + num_skipped_cycles:
                        # Get the absolute start and end indices for the data slice
                        start_idx_abs = heelstrike_indices_R[-3]; end_idx_abs = heelstrike_indices_R[-1]
                        print('\nR', start_idx_abs, heelstrike_indices_R[-2:-1], end_idx_abs)
                        mid_peak_idx_rel = heelstrike_indices_R[-2:-1] - start_idx_abs # This is relative about start_idx_abs

                        # Slice the data from the input stream buffer
                        start_idx_rel = start_idx_abs - buffer_start_abs; end_idx_rel = end_idx_abs - buffer_start_abs
                        input_stream_data_R = input_stream_data[1, :, start_idx_rel:end_idx_rel]

                        self.online_adaptator.trigger_finetuning('R', current_incline, current_speed, input_stream_data_R.T.copy(), mid_peak_idx_rel, loop_index)

                        # Update the last used peak to the end of the current window
                        self.last_used_peak_idx_R = end_idx_abs

            # Check for new weights from the adaptation worker
            updated_params = self.online_adaptator.get_updated_weights()
            if updated_params:
                side, input_reduced, grid_key, start_index, weights, biases = updated_params
                if side == 'R':
                    self.linear_weights_R = weights;    self.linear_biases_R = biases
                    start_idx_R = start_index;      log_input_reduced_R[loop_index] = input_reduced; log_grid_key_R[loop_index] = grid_key
                    update_latency_R = (loop_index - start_index) / self.Exo.control_freq_Hz
                    print(f'Update latency (R): {update_latency_R:.2f} sec')
                elif side == 'L':
                    self.linear_weights_L = weights;    self.linear_biases_L = biases
                    start_idx_L = start_index;      log_input_reduced_L[loop_index] = input_reduced; log_grid_key_L[loop_index] = grid_key
                    update_latency_L = (loop_index - start_index) / self.Exo.control_freq_Hz
                    print(f'Update latency (L): {update_latency_L:.2f} sec')

            # 5. TensorRT inference & Apply linear layer weights and biases
            if self.gait_phase_output_q.empty():
                self.gait_phase_input_q.put((model_input_arr[0, :, :].copy(), model_input_arr[1, :, :].copy(), loop_index))
            
            try:
                model_output_l_val, model_output_r_val, gp_loop_index = self.gait_phase_output_q.get_nowait()
                last_model_output_l, last_model_output_r = model_output_l_val, model_output_r_val
            except mp.queues.Empty:
                model_output_l_val, model_output_r_val = last_model_output_l, last_model_output_r
            
            # Apply linear layer weights and biases
            model_output_l_val = np.dot(model_output_l_val.flatten(), self.linear_weights_L.T)
            model_output_l_val += self.linear_biases_L
            model_output_r_val = np.dot(model_output_r_val.flatten(), self.linear_weights_R.T)
            model_output_r_val += self.linear_biases_R

            # 6. Calculate the gait phase
            model_output_l_denorm = model_output_l_val * self.label_std + self.label_mean
            model_output_r_denorm = model_output_r_val * self.label_std + self.label_mean
            gait_phase_L, gait_phase_R = cartesian_to_percentage(model_output_l_denorm), cartesian_to_percentage(model_output_r_denorm)

            # Store previous gait phase if decreasing 
            if (3 < gait_phase_L_prev < 93) and (gait_phase_L < gait_phase_L_prev): gait_phase_L = gait_phase_L_prev # handle the case that decreases suddenly
            elif (gait_phase_L_prev <= 5) and (gait_phase_L > 75): gait_phase_L = gait_phase_L_prev # handle the case that suddenly jump back (from low to high)
            elif (gait_phase_L - gait_phase_L_prev) > 30: gait_phase_L = gait_phase_L_prev  # handle the case that suddenly jump forward
            else: gait_phase_L_prev = gait_phase_L # update normally
            if (3 < gait_phase_R_prev < 93) and (gait_phase_R < gait_phase_R_prev): gait_phase_R = gait_phase_R_prev
            elif (gait_phase_R_prev <= 5) and (gait_phase_R > 75): gait_phase_R = gait_phase_R_prev
            elif (gait_phase_R - gait_phase_R_prev) > 30: gait_phase_R = gait_phase_R_prev
            else: gait_phase_R_prev = gait_phase_R

            # Calculate gradual torque scaling factor
            gradual_torque_scale = min(1.0, ((loop_index / self.Exo.control_freq_Hz)) / self.adjustment_duration)

            # 7. Send the torque command to the motors
            if gait_phase_L < 3:
                motor_cmd_val_L = self.torque_profile[current_incline][current_speed][int(gait_phase_L)] * self.body_mass_kg * self.Exo.scale_factor * gradual_torque_scale
                prev_incline = current_incline; prev_speed = current_speed
            else:
                motor_cmd_val_L = self.torque_profile[prev_incline][prev_speed][int(gait_phase_L)] * self.body_mass_kg * self.Exo.scale_factor * gradual_torque_scale
            
            if gait_phase_R < 3:
                motor_cmd_val_R = self.torque_profile[current_incline][current_speed][int(gait_phase_R)] * self.body_mass_kg * self.Exo.scale_factor * gradual_torque_scale
                prev_incline = current_incline; prev_speed = current_speed
            else:
                motor_cmd_val_R = self.torque_profile[prev_incline][prev_speed][int(gait_phase_R)] * self.body_mass_kg * self.Exo.scale_factor * gradual_torque_scale

            motor_cmd_array = fast_roll(motor_cmd_array)
            motor_cmd_array[:, -1] = [motor_cmd_val_R, motor_cmd_val_L]    

            if Exo_ON == False: motor_cmd_val_L, motor_cmd_val_R = 0.0, 0.0 # use this for Exo off condition
            if motor_cmd_val_L > self.Exo.max_torque:    motor_cmd_val_L = self.Exo.max_torque 
            elif motor_cmd_val_L < -self.Exo.max_torque: motor_cmd_val_L = -self.Exo.max_torque
            if motor_cmd_val_R > self.Exo.max_torque:    motor_cmd_val_R = self.Exo.max_torque
            elif motor_cmd_val_R < -self.Exo.max_torque: motor_cmd_val_R = -self.Exo.max_torque

            self.Exo.mtr_comms.set_torque(self.Exo.CAN_id_L, motor_cmd_val_L) 
            self.Exo.mtr_comms.set_torque(self.Exo.CAN_id_R, -motor_cmd_val_R) # Negative sign because the motor is mounted in reverse direction

            # 8. Stack the data (that will be saved after the trial)
            log_mtr_cmd_L[loop_index], log_mtr_cmd_R[loop_index] = motor_cmd_val_L, motor_cmd_val_R
            log_gait_phase_L[loop_index], log_gait_phase_R[loop_index] = gait_phase_L, gait_phase_R

            # GPIO pulse logic starting from here
            current_time = time.time() - start_time
            
            # First pulse
            if current_time >= (self.pulse_after_start) and not first_pulse_sent:
                self.GPIO_control.send_gpio_pulse_start()
                first_pulse_sent = True
                first_pulse_end_time = current_time + 0.2  # 200ms pulse duration
            # First pulse end
            if first_pulse_sent and first_pulse_end_time and current_time >= first_pulse_end_time:
                self.GPIO_control.send_gpio_pulse_end()
                first_pulse_end_time = None            
            # Second pulse
            if current_time >= (self.pulse_after_start + self.trial_dur_sec) and not second_pulse_sent:
                self.GPIO_control.send_gpio_pulse_start()
                second_pulse_sent = True
                second_pulse_end_time = current_time + 0.2  # 200ms pulse duration
            # Second pulse end
            if second_pulse_sent and second_pulse_end_time and current_time >= second_pulse_end_time:
                self.GPIO_control.send_gpio_pulse_end()
                second_pulse_end_time = None
                break # Exit the loop after the second pulse ends

            # GPIO output logging
            log_gpio_output[loop_index] = self.GPIO_control.get_gpio_output_state()

            # 9. Loop time
            loop_time_exceeded = (time.time() - start_time) - (loop_index / self.Exo.control_freq_Hz)

            # 10. Send telemetry data
            telemetry_data = {
                "pos_L": mtr_pos_L,
                "pos_R": mtr_pos_R,
                # "gyroY_L": imu_L[4],
                # "gyroY_R": imu_R[4],
                "GRF_L": GRF_L,
                "GRF_R": GRF_R,
                "gait_phase_L": gait_phase_L,
                "gait_phase_R": gait_phase_R,
                "cmd_L": motor_cmd_val_L,
                "cmd_R": motor_cmd_val_R,
                "incline": current_incline,
                "speed": current_speed,
                "update_latency_L": update_latency_L,
                "update_latency_R": update_latency_R,
                "start_idx_L": start_idx_L,
                "start_idx_R": start_idx_R,
                "loop_time_exceeded": loop_time_exceeded,
                "gp_inference": (gp_loop_index-loop_index),
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
        save_weights_biases(self.linear_weights_L, self.linear_biases_L, self.linear_weights_R, self.linear_biases_R, self.linear_layer_path)
        cleanup_can(self.Exo.bus, self.Exo.notifier)
        self.GPIO_control.safe_gpio_cleanup()

        gc.collect()
        torch.cuda.empty_cache()

        print("Exiting program")
        os._exit(0)