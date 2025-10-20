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
    pulse_after_start = 2 # seconds
    trial_dur_sec = 120  # seconds
    exo_ON = True

    # Trigger setting
    trigger_type = "typing"  # "mocap" or "typing"

    # Body mass setting
    body_mass_kg = 72  # kg

    # Model path
    # tcn_only
    trt_engine_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/hyperparam_optimized-input_modality_7/hyperparam_optimized-input_modality_7_tcn.trt'
    # entire model (TCN + linear layer)
    pt_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/hyperparam_optimized-input_modality_7/hyperparam_optimized-input_modality_7.pt'
    # torque profile path
    torque_profile_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE/avg_biological_hip_torque.pkl'

    # Initialize Control loop class
    controller = Controller(pt_model_path, trt_engine_path, torque_profile_path, trigger_type, trial_name,
                            pulse_after_start, trial_dur_sec)
    
    if controller.trigger_type == "mocap":
        mocap_trigger = Mocap_trigger()
        mocap_trigger.start_client()

    mp.set_start_method('spawn', force=True)

    controller.run_loop(exo_ON)