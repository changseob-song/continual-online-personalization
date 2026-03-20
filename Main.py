import torch, gc
import multiprocessing as mp
from Controller import Controller

if __name__ == '__main__':

    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Trial setting'
    subject = 'SK_Amy'
    condition = 'task_buffer'
    course_num = 5 # 1, 2, 3, 4, 5, 6
    incline = 'RD_slow' # LG, RA_5, RD_5, LG, RA_5, RD_5

    trial_name = f'{subject}-{condition}-{course_num}_{incline}'

    pulse_after_start = 10  # seconds
    trial_dur_sec = 60 * .5  # seconds
    adjustment_duration = 10  # seconds
    exo_ON = True
    adaptation_ON = True
    replay_buffer_ON = True
    PC_USE = False

    # Trigger setting
    trigger_type = "mocap"  # "mocap" or "typing"

    # Body mass setting
    body_mass_kg = 54  # kg

    # Model path
    # tcn_only
    trt_engine_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/motor_bilateral_pos/motor_bilateral_pos.trt'
    # entire model (TCN + linear layer)
    pt_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/trained_model/motor_bilateral_pos/motor_bilateral_pos.pt'
    # torque profile path
    torque_profile_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/torque_splines.pkl'
    # PCA model path
    pca_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/pca_model_pos_1gc_2_50_outlierexcluded.pkl'
    # Encoder model path
    encoder_model_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/autoencoder_mtr.pt'

    # Linear layer numpy file
    linear_layer_path = f'/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/linear_layer_weights_biases_{course_num-1}.pkl'
    # Buffer file
    buffer_file_path = f'/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/buffer_{course_num-1}.pkl'
    
    # Initialize Control loop class
    controller = Controller(pt_model_path, trt_engine_path, torque_profile_path, pca_model_path,
                            encoder_model_path, linear_layer_path, buffer_file_path,
                            trigger_type, trial_name, course_num, incline,
                            pulse_after_start, trial_dur_sec, adjustment_duration,
                            body_mass_kg,
                            adaptation_ON, replay_buffer_ON, PC_USE
                            )

    mp.set_start_method('spawn', force=True)

    controller.run_loop(exo_ON)