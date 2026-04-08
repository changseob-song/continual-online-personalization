import torch, gc, os, time
import multiprocessing as mp
from Controller import Controller
from Exo import Exo
from Utils_Mocap_Datastream import Mocap_trigger

if __name__ == '__main__':

    # Garbage collection & clearing GPU cache
    gc.collect()
    torch.cuda.empty_cache()
    # os.system("sudo sync; sudo sysctl -w vm.drop_caches=3")
    os.system("fuser -k -9 /dev/nvhost-gpu >/dev/null 2>&1")
    os.system("fuser -k -9 /dev/nvhost-ctrl-gpu >/dev/null 2>&1")
    time.sleep(1)
    
    # Trial setting
    subject = 'SK_Amy'
    condition = 'pc_buffer'

    # Condition
    exo_ON = True
    adaptation_ON = True
    replay_buffer_ON = True
    PC_USE = True

    courses = {
        # '1_1': 'LG_slow', 
        # '2_1': 'LG_fast',
        # '3_1': 'RA_slow',
        # '4_1': 'RA_fast',
        # '5_1': 'RD_slow',
        # '6_1': 'RD_fast',
        # '7_1': 'LG_slow',
        # '8_1': 'LG_fast',
        '8_2': 'RA_slow',
        '8_3': 'RA_fast',
        '8_4': 'RD_slow',
        '8_5': 'RD_fast',
        '7_2': 'LG_fast',
        '8_6': 'LG_slow',
        '8_7': 'RA_slow',
        '8_8': 'RA_fast',
        '8_9': 'RD_slow',
        '8_10': 'RD_fast'
    }

    # Body mass setting
    body_mass_kg = 52  # kg
    # Trigger setting
    trigger_type = "mocap"  # "mocap" or "typing"
    # Exo setting
    Exo = Exo()

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
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    for course_num, task in courses.items():

        course_num_ = int(str(course_num)[0])
        incline = task[:2]

        trial_name = f'{subject}-{condition}-{str(course_num_)}_{str(course_num)[2]}_{task}'
        print(f"\nStarting trial: {trial_name}")

        pulse_after_start = 10  # seconds
        adjustment_duration = 10  # seconds
        if course_num_ == 7:
            trial_dur_sec = 60 * 5  # seconds
        else:
            trial_dur_sec = 60 * .5  # seconds

        # Linear layer numpy file
        linear_layer_path = f'/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/linear_layer_weights_biases_{course_num_-1}.pkl'
        # Buffer file
        buffer_file_path = f'/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/buffer_{course_num_-1}.pkl'
        
        # Initialize Control loop class
        controller = Controller(Exo, pt_model_path, trt_engine_path, torque_profile_path, pca_model_path,
                                encoder_model_path, linear_layer_path, buffer_file_path,
                                trigger_type, trial_name, course_num_, incline,
                                pulse_after_start, trial_dur_sec, adjustment_duration,
                                body_mass_kg,
                                adaptation_ON, replay_buffer_ON, PC_USE
                                )

        controller.run_loop(exo_ON)