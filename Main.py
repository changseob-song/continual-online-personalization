import torch, gc
import multiprocessing as mp

from Controller import Controller
from Utils_Mocap_trigger import Mocap_trigger

if __name__ == '__main__':

    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Trial setting
    trial_name = 'online_test_5-adaptation_on'
    pulse_after_start = 0 # seconds
    trial_dur_sec = 90  # seconds
    adjustment_duration = 5  # seconds
    exo_ON = True
    adaptation_ON = True
    replay_buffer_ON = False

    # Trigger setting
    trigger_type = "typing"  # "mocap" or "typing"

    # Body mass setting
    body_mass_kg = 72  # kg

    # Task stream
    task_stream = ['LG-1p0mps', 'LG-1p0mps', 'LG-1p0mps',
                   'RA_5deg-0p5mps', 'RA_5deg-0p8mps',
                   'LG-0p6mps', 'LG-0p8mps',
                   'RD_10deg-0p8mps', 'RD_10deg-1p0mps',
                   'LG-0p5mps', 'LG-0p7mps']
    task_interval = 30 # seconds

    # Model path
    # tcn_only
    trt_engine_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/hyperparam_optimized-input_modality_6/hyperparam_optimized-input_modality_6_tcn.trt'
    # entire model (TCN + linear layer)
    pt_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/hyperparam_optimized-input_modality_6/hyperparam_optimized-input_modality_6.pt'
    # torque profile path
    torque_profile_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE/torque_splines.pkl'
    # AB average input path
    ab_avg_input_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE/ab_avg_input.pkl'

    # Initialize Control loop class
    controller = Controller(pt_model_path, trt_engine_path, torque_profile_path, ab_avg_input_path,
                            trigger_type, trial_name,
                            pulse_after_start, trial_dur_sec, adjustment_duration,
                            body_mass_kg,
                            task_stream, task_interval,
                            replay_buffer_ON
                            )

    mp.set_start_method('spawn', force=True)

    controller.run_loop(exo_ON, adaptation_ON)