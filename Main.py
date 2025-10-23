import torch, gc
import multiprocessing as mp

from Controller import Controller
from Utils_Mocap_trigger import Mocap_trigger

if __name__ == '__main__':

    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Trial setting
    trial_name = 'debug'
    pulse_after_start = 0 # seconds
    trial_dur_sec = 60  # seconds
    exo_ON = True

    # Trigger setting
    trigger_type = "typing"  # "mocap" or "typing"

    # Body mass setting
    body_mass_kg = 72  # kg

    # Task stream
    task_stream = ['LG-1p0mps', 'LG-0p4mps',
                   'RA_5deg-0p5mps', 'RA_5deg-0p8mps',
                   'LG-0p6mps', 'LG-0p8mps',
                   'RD_10deg-0p8mps', 'RD_10deg-1p0mps',
                   'LG-0p5mps', 'LG-0p7mps']
    task_interval = 30 # seconds

    # Model path
    # tcn_only
    trt_engine_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/hyperparam_optimized-input_modality_7/hyperparam_optimized-input_modality_7_tcn.trt'
    # entire model (TCN + linear layer)
    pt_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/hyperparam_optimized-input_modality_7/hyperparam_optimized-input_modality_7.pt'
    # torque profile path
    torque_profile_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE/avg_biological_hip_torque.pkl'

    # Initialize Control loop class
    controller = Controller(pt_model_path, trt_engine_path, torque_profile_path,
                            trigger_type, trial_name,
                            pulse_after_start, trial_dur_sec,
                            body_mass_kg,
                            task_stream, task_interval
                            )

    mp.set_start_method('spawn', force=True)

    controller.run_loop(exo_ON)