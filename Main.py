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
    trial_start_sec = 2  # seconds
    trial_dur_sec = 120  # seconds
    pulse_after_start = 2 # seconds
    exo_ON = True

    # Online adaptation setting
    update_interval_sec = 5  # seconds

    # Trigger setting
    trigger_type = "typing"  # "mocap" or "typing"

    # Body mass setting
    body_mass_kg = 72  # kg

    # Model path
    # tcn_only
    trt_engine_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/hyperparam_optimized-all_subjects/hyperparam_optimized-all_subjects_tcn.trt'
    # entire model (TCN + linear layer)
    pt_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/hyperparam_optimized-all_subjects/hyperparam_optimized-all_subjects.pt'
    # torque profile path
    torque_profile_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE/avg_biological_hip_torque.pkl'

    # Initialize Control loop class
    controller = Controller(pt_model_path, trt_engine_path, torque_profile_path, trigger_type, trial_name,
                            trial_start_sec, trial_dur_sec, pulse_after_start,
                            update_interval_sec,)
    
    if controller.trigger_type == "mocap":
        mocap_trigger = Mocap_trigger()
        mocap_trigger.start_client()

    mp.set_start_method('spawn', force=True)

    controller.run_loop(exo_ON)