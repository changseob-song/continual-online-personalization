import torch, gc
import multiprocessing as mp
from Controller import Controller

if __name__ == '__main__':

    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Trial setting'
    subject = 'SK_Maria'
    condition = 'adapted'
    course_num = 2
    incline = 'LG'

    trial_name = f'pilot-{subject}-{condition}-{course_num}_{incline}'

    pulse_after_start = 10  # seconds
    trial_dur_sec = 4 * 60 + 10  # seconds
    adjustment_duration = 10  # seconds
    exo_ON = False
    adaptation_ON = True
    replay_buffer_ON = False

    # Trigger setting
    trigger_type = "mocap"  # "mocap" or "typing"

    # Body mass setting
    body_mass_kg = 72  # kg

    # Model path
    # tcn_only
    trt_engine_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/heel_strike-RD_flipped-bilateral-transfer/heel_strike-RD_flipped-bilateral-transfer.trt'
    # entire model (TCN + linear layer)
    pt_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/heel_strike-RD_flipped-bilateral-transfer/heel_strike-RD_flipped-bilateral-transfer.pt'
    # Linear layer numpy file
    linear_layer_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/linear_layer_weights_biases.pkl'
    # torque profile path
    torque_profile_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/torque_splines.pkl'
    
    # Initialize Control loop class
    controller = Controller(pt_model_path, trt_engine_path, linear_layer_path, torque_profile_path,
                            trigger_type, trial_name, course_num, incline,
                            pulse_after_start, trial_dur_sec, adjustment_duration,
                            body_mass_kg,
                            adaptation_ON, replay_buffer_ON
                            )

    mp.set_start_method('spawn', force=True)

    controller.run_loop(exo_ON)