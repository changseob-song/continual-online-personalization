import torch, gc
import multiprocessing as mp
from Controller import Controller
from Utils_Mocap_trigger import Mocap_trigger

if __name__ == '__main__':

    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Trial setting'
    trial_name = 'test3_exo_off'
    pulse_after_start = 0  # seconds
    trial_dur_sec = 60*1 #15*2  # seconds
    adjustment_duration = 10  # seconds
    exo_ON = False
    adaptation_ON = True
    replay_buffer_ON = False

    # Trigger setting
    trigger_type = "typing"  # "mocap" or "typing"

    # Body mass setting
    body_mass_kg = 72  # kg

    # Model path
    # tcn_only
    trt_engine_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/heel_strike-test/heel_strike-test_tcn.trt'
    # entire model (TCN + linear layer)
    pt_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/heel_strike-test/heel_strike-test.pt'
    # task estimator path
    trt_task_estimator_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/task_estimator-input_modality_6_window_size_200/task_estimator-input_modality_6.trt'
    # torque profile path
    torque_profile_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE/torque_splines.pkl'
    
    # Initialize Control loop class
    controller = Controller(pt_model_path, trt_engine_path, trt_task_estimator_path, torque_profile_path,
                            trigger_type, trial_name,
                            pulse_after_start, trial_dur_sec, adjustment_duration,
                            body_mass_kg,
                            adaptation_ON, replay_buffer_ON
                            )

    mp.set_start_method('spawn', force=True)

    controller.run_loop(exo_ON)