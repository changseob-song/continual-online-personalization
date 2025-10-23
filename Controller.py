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
    def __init__(self, pt_model_path, trt_engine_path, torque_profile_path,
                 trigger_type, trial_name, pulse_after_start, trial_dur_sec, body_mass_kg,
                 task_stream, task_interval):
        self.pt_model_path = pt_model_path
        self.pt_model_linear_path = pt_model_path.replace('.pt', '_linear.pt')
        self.trt_engine_path = trt_engine_path
        self.trigger_type = trigger_type
        self.trial_name = trial_name
        self.body_mass_kg = body_mass_kg
        self.pulse_after_start = pulse_after_start
        self.trial_dur_sec = trial_dur_sec
        self.task_stream = task_stream
        self.task_interval = task_interval
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
            'imu_P': np.zeros((max_samples, 6)), 'imu_L': np.zeros((max_samples, 6)), 'imu_R': np.zeros((max_samples, 6)),
            'mtr_cmd_L': np.zeros(max_samples), 'mtr_cmd_R': np.zeros(max_samples),
            'gait_phase_L': np.zeros(max_samples), 'gait_phase_R': np.zeros(max_samples),
            'incline': ['']*max_samples, 'speed': ['']*max_samples,
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

        self.input_mean = np.load(input_mean_path); self.input_std = np.load(input_std_path)
        self.label_mean = np.load(label_mean_path); self.label_std = np.load(label_std_path)
        self.num_input_features = self.input_mean.shape[0]

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
                                    args=(self.input_q, self.output_q, self.trt_engine_path,
                                            input_mean_path, input_std_path,label_mean_path, label_std_path,
                                            self.num_input_features, self.Exo.frame_length))
        self.inference_process.start()

        # Extract linear layer weights and biases from the PyTorch model
        state_dict = torch.load(self.pt_model_linear_path, map_location="cpu", weights_only=True)
        self.linear_weights_R = state_dict['weight'].numpy().astype(np.float32)
        self.linear_biases_R = state_dict['bias'].numpy().astype(np.float32)
        self.linear_weights_L = self.linear_weights_R.copy()
        self.linear_biases_L = self.linear_biases_R.copy()

        # Initialize OnlineAdaptator
        self.online_adaptator = OnlineAdaptator(self.pt_model_path)

    def run_loop(self, Exo_ON=False):

        # Setting for the exiting process
        atexit.register(lambda: (cleanup_can(self.Exo.bus, self.Exo.notifier), self.GPIO_control.safe_gpio_cleanup()))
        signal.signal(signal.SIGINT, self.exit_signal_handler)

        # Rolling array initialization of model output array (2xframe_length)
        model_input_arr = np.zeros((2, self.num_input_features, self.Exo.frame_length), dtype=np.float32)
        # Predefine input stream data size for online adaptation (6 seconds buffer)
        input_stream_data = np.zeros((2, self.num_input_features, 6 * self.Exo.control_freq_Hz), dtype=np.float32)  # 100 frames of input data
        motor_cmd_array = np.zeros((2, self.Exo.frame_length), dtype=np.float32)  # for torque command filtering

        current_pos_L, current_vel_L = 0.0, 0.0
        current_pos_R, current_vel_R = 0.0, 0.0

        local_p_data = np.zeros(6); local_l_data = np.zeros(6); local_r_data = np.zeros(6)

        right_data = np.zeros(self.num_input_features); left_data = np.zeros(self.num_input_features)

        last_model_output_r = np.zeros((80, 100), dtype=np.float32); last_model_output_l = np.zeros((80, 100), dtype=np.float32)
        model_output_r_val = last_model_output_r; model_output_l_val = last_model_output_l

        update_time_R, update_time_L = 0.0, 0.0

        # Create local references to data arrays for faster access
        log_timestamp = self.data_to_save['timestamp']
        log_mtr_pos_L, log_mtr_pos_R = self.data_to_save['mtr_pos_L'], self.data_to_save['mtr_pos_R']
        log_mtr_vel_L, log_mtr_vel_R = self.data_to_save['mtr_vel_L'], self.data_to_save['mtr_vel_R']
        log_imu_P, log_imu_L, log_imu_R = self.data_to_save['imu_P'], self.data_to_save['imu_L'], self.data_to_save['imu_R']
        log_mtr_cmd_L, log_mtr_cmd_R = self.data_to_save['mtr_cmd_L'], self.data_to_save['mtr_cmd_R']
        log_gait_phase_L, log_gait_phase_R = self.data_to_save['gait_phase_L'], self.data_to_save['gait_phase_R']
        log_incline, log_speed = self.data_to_save['incline'], self.data_to_save['speed']
        log_gpio_output = self.data_to_save['gpio_output']

        # Start recording time
        logging_started = False
        first_pulse_sent = False
        first_pulse_end_time = None
        second_pulse_sent = False
        second_pulse_end_time = None
        loop_index = 1

        # Wait for the trigger to start the trial
        if self.trigger_type == "mocap":
            print("Wait for the tensorrt to warm up...\n")
        elif self.trigger_type == "typing":
            input_trigger = input("Wait for the tensorrt to warm up...\n")
            if input_trigger == "":
                print("Trial started")

        # Main control loop
        while True:

            # 0. Check if the trial time has exceeded
            start_time = time.time()

            if self.trigger_type == "mocap" and not logging_started:
                self.mocap_trigger.wait_for_trigger()
                print("Mocap trigger received - starting data logging")
                start_time = time.time()
                logging_started = True
            elif self.trigger_type == "typing" and not logging_started:
                start_time = time.time()
                logging_started = True

            # Get the task index
            task_idx = int((loop_index/self.Exo.control_freq_Hz - self.pulse_after_start)//self.task_interval)
            task_idx = max(0, task_idx)
            current_incline, current_speed = self.task_stream[task_idx].split('-')
            log_incline[loop_index] = current_incline; log_speed[loop_index] = current_speed

            # 1. Read the motor encoder values
            current_pos_L, current_vel_L = self.Exo.update_readings(self.Exo.CAN_id_L)
            current_pos_R, current_vel_R = self.Exo.update_readings(self.Exo.CAN_id_R)

            current_pos_R *= -1; current_vel_R *= -1 # mirror the right side values (because of the motor mounting direction)

            log_mtr_pos_L[loop_index] = current_pos_L; log_mtr_pos_R[loop_index] = current_pos_R
            log_mtr_vel_L[loop_index] = current_vel_L; log_mtr_vel_R[loop_index] = current_vel_R

            # 2. Read the IMU values
            imu_dict = self.Exo.imus.read_IMUs()

            local_p_data = imu_dict["IMU_PELVIS"]; local_l_data = imu_dict["IMU_THIGH_LEFT"]; local_r_data = imu_dict["IMU_THIGH_RIGHT"]
            log_imu_P[loop_index, :] = local_p_data; log_imu_L[loop_index, :] = local_l_data; log_imu_R[loop_index, :] = local_r_data

            # 3. Mirror the left data to the right side
            p_data_reflected = local_p_data.copy()
            p_data_reflected[1] *= -1; p_data_reflected[3] *= -1; p_data_reflected[5] *= -1

            l_data_reflected = local_l_data.copy()
            l_data_reflected[1] *= -1; l_data_reflected[3] *= -1; l_data_reflected[5] *= -1
            
            # 4. Prepare the model input data
            right_data[:6] = local_r_data; right_data[6] = current_pos_R
            left_data[:6] = l_data_reflected; left_data[6] = current_pos_L

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

                peak_indices_L, _ = find_peaks(-recent_pos_L, height=None, distance=15, prominence=10)
                peak_indices_L += search_start_idx # This makes the indices back to absolute timeframe
                peak_indices_R, _ = find_peaks(-recent_pos_R, height=None, distance=15, prominence=10)
                peak_indices_R += search_start_idx
                
                buffer_start_abs = loop_index - len(input_stream_data[0, 0, :])

                # Check if there are enough new peaks for an update (2 gait cycles = 2 new peaks after the start)
                if self.last_used_peak_idx_R not in peak_indices_R:
                    if len(peak_indices_R) > update_freq_gc:
                        # Get the absolute start and end indices for the data slice
                        start_idx_abs = peak_indices_R[0]; end_idx_abs = peak_indices_R[-1]
                        print('R', start_idx_abs, peak_indices_R[-2], end_idx_abs)
                        mid_peak_idx_rel = peak_indices_R[1:-1] - start_idx_abs # This is relative about start_idx_abs
                        incline_stream = log_incline[start_idx_abs:end_idx_abs]; speed_stream = log_speed[start_idx_abs:end_idx_abs]

                        # Slice the data from the input stream buffer
                        start_idx_rel = start_idx_abs - buffer_start_abs; end_idx_rel = end_idx_abs - buffer_start_abs
                        input_stream_data_R = input_stream_data[0, :, start_idx_rel:end_idx_rel]

                        self.online_adaptator.trigger_finetuning('R', current_incline, current_speed, input_stream_data_R.T.copy(), mid_peak_idx_rel)

                        # Update the last used peak to the end of the current window
                        self.last_used_peak_idx_R = end_idx_abs

                # Check if there are enough new peaks for an update (2 gait cycles = 2 new peaks after the start)
                if self.last_used_peak_idx_L not in peak_indices_L:
                    if len(peak_indices_L) > update_freq_gc:
                        # Get the absolute start and end indices for the data slice
                        start_idx_abs = peak_indices_L[0]; end_idx_abs = peak_indices_L[-1]
                        mid_peak_idx_rel = peak_indices_L[1:-1] - start_idx_abs # This is relative about start_idx_abs
                        incline_stream = log_incline[start_idx_abs:end_idx_abs]; speed_stream = log_speed[start_idx_abs:end_idx_abs]

                        # Slice the data from the input stream buffer
                        start_idx_rel = start_idx_abs - buffer_start_abs; end_idx_rel = end_idx_abs - buffer_start_abs
                        input_stream_data_L = input_stream_data[1, :, start_idx_rel:end_idx_rel]

                        self.online_adaptator.trigger_finetuning('L', current_incline, current_speed, input_stream_data_L.T.copy(), mid_peak_idx_rel)

                        # Update the last used peak to the end of the current window
                        self.last_used_peak_idx_L = end_idx_abs

            # Check for new weights from the adaptation worker
            updated_params = self.online_adaptator.get_updated_weights()
            if updated_params:
                side, update_time, weights, biases = updated_params                
                if side == 'R':
                    self.linear_weights_R = weights
                    self.linear_biases_R = biases
                    update_time_R = update_time
                elif side == 'L':
                    self.linear_weights_L = weights
                    self.linear_biases_L = biases
                    update_time_L = update_time

            # 5. TensorRT inference & Apply linear layer weights and biases
            self.input_q.put((model_input_arr[0, :, :].copy(), model_input_arr[1, :, :].copy()))
            try:
                model_output_r_val, model_output_l_val = self.output_q.get_nowait()
                last_model_output_r, last_model_output_l = model_output_r_val, model_output_l_val
            except mp.queues.Empty:
                model_output_r_val, model_output_l_val = last_model_output_r, last_model_output_l
            
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

            delayed_gait_phase_R = (gait_phase_R - self.Exo.delay_factor) % 100
            delayed_gait_phase_L = (gait_phase_L - self.Exo.delay_factor) % 100

            # 7. Send the torque command to the motors
            motor_cmd_val_L = self.torque_profile[current_incline][current_speed][int(delayed_gait_phase_L)] * self.body_mass_kg * self.Exo.scale_factor
            motor_cmd_val_R = self.torque_profile[current_incline][current_speed][int(delayed_gait_phase_R)] * self.body_mass_kg * self.Exo.scale_factor

            motor_cmd_array = fast_roll(motor_cmd_array)
            motor_cmd_array[:, -1] = [motor_cmd_val_R, motor_cmd_val_L]    

            # 8. Filter the torque command
            motor_cmd_val_L = causal_filter(motor_cmd_array[1, :], tau=0.05)[-1]
            motor_cmd_val_R = causal_filter(motor_cmd_array[0, :], tau=0.05)[-1]

            if Exo_ON == False: motor_cmd_val_L, motor_cmd_val_R = 0.0, 0.0 # use this for Exo off condition

            if motor_cmd_val_L > 10 or motor_cmd_val_R > 10:
                motor_cmd_val_L, motor_cmd_val_R = 10, 10
            if motor_cmd_val_L < -10 or motor_cmd_val_R < -10:
                motor_cmd_val_L, motor_cmd_val_R = -10, -10

            self.Exo.mtr_comms.set_torque(self.Exo.CAN_id_L, -motor_cmd_val_L) # Negative sign because the motor is mounted in reverse direction
            self.Exo.mtr_comms.set_torque(self.Exo.CAN_id_R, motor_cmd_val_R)

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

            # GPIO output logging
            log_gpio_output[loop_index] = self.GPIO_control.get_gpio_output_state()

            # 9. Loop time
            loop_time_exceeded = (time.time() - start_time) - (loop_index / self.Exo.control_freq_Hz)

            # 10. Send telemetry data
            telemetry_data = {
                "pos_R": current_pos_R,
                "pos_L": current_pos_L,
                "cmd_R": motor_cmd_val_R,
                "cmd_L": motor_cmd_val_L,
                "gait_phase_L": gait_phase_L,
                "gait_phase_R": gait_phase_R,
                "update_time_R": update_time_R,
                "update_time_L": update_time_L,
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