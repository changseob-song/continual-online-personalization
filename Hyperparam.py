# ICORR_TCN_Training.py
import torch
import os

# from Model import TCN


use_sweep = False  # Set to True to run sweep, False for single run
# Define the sweep configuration outside the main function
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'RMSE',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {'values': [8, 16, 32, 64, 128]}, # Different batch sizes for TCN
    } # NOTE: change the save_sub_dir to save results in different directories
}

# Base hyperparameters
hyperparam_config = {
    'wandb_project_name': 'online_adaptation-GPE',
    'wandb_session_name': 'hyperparam_optimized-all_subjects',  # sweep-, 2. SK, 3. AB+SK
    'input_size': 2, # 12 for IMU (right, left), 2 for hip angle and velocity
    'output_size': 2, # 2 for polar coordinates (x, y) of gait cycle
    'architecture': 'TCN',
    
    'transfer_learning': False,
    'dataset_proportion': 1.0, # dataset proportion for training
    
    'epochs': 50,
    'validation_split': 0.1,
    'number_of_layers': 2,
    'dilations': [1, 2, 4, 8, 16],

    'window_size': 100,
    'num_channels': [80, 80, 80, 80, 80],
    'kernel_size': 5,
    'dropout': 0.05,
    'init_lr': 1e-4,
    'batch_size': 16,

    'number_of_workers': 10,
}

def train():
    # wandb login
    os.environ["WANDB_API_KEY"] = "9d7294877630de627f0413ca39aebd4c9a387e50"
    if use_sweep:
        wandb_run = wandb.init(config=hyperparam_config, name=hyperparam_config['wandb_session_name'])
    else:
        wandb_run = wandb.init(config=hyperparam_config, project=hyperparam_config['wandb_project_name'], name=hyperparam_config['wandb_session_name'])

    # Access wandb.config after initializing wandb
    config = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Update hyperparameters with wandb config (useful if overridden)
    hyperparam_config.update(dict(config))

    # Create directory for results & plots
    save_dir = '/home/metamobility3/Changseob/online_learning-non_causal/training_result-gpe_adaptation'

    if use_sweep:
        sweep_target_param = list(sweep_config['parameters'].keys())[0]
        save_sub_dir = f"{hyperparam_config['wandb_session_name']}_{sweep_target_param}_{config[sweep_target_param]}"
    else:
        save_sub_dir = hyperparam_config['wandb_session_name']

    save_dir = os.path.join(save_dir, save_sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Model Initialization
    model = TCN(hyperparam_config).to(device)
    
    # Load pretrained model if transfer learning is enabled
    if hyperparam_config['transfer_learning']:
        pretrained_model_path = '/home/metamobility3/Changseob/biotorque/training_result/baseline_TCN'
        model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'baseline_TCN.pt'), map_location=device))
        print("\nPretrained model loaded: ", pretrained_model_path)
        #Freeze the TCN part of the model
        for param in model.tcn.parameters():
            param.requires_grad = False
    else:
        pretrained_model_path = None

    # Initialize DataHandler
    data_root = '/home/metamobility3/Changseob/dataset_MeMo/Synced_Biotorque_LG_Data'
    data_handler = DataHandler(data_root, hyperparam_config, pretrained_model_path)
    data_handler.load_data(
        train_data_partition=[
                        'AB01_Jimin',
                        'AB02_Rajiv',
                        'AB03_Amy',
                        'AB04_Changseob',
                        'AB05_Maria',
                        'AB06_Vaidehi',
                        'AB07_Leo',
                        'AB08_Adrian',
                        'AB09_Crystal',
                        'AB10_Pragya',
                        'AB11_Ryan',
                        'AB12_Ray',
                        'AB13_Hridayam',
                        'AB14_Evy',
        ],
        train_data_condition=[
                        #'0mps', 
                        '0p2mps', '0p4mps', '0p6mps', '0p8mps', '1p0mps', '1p2mps', '1p4mps', 'transient_15sec', 'transient_30sec',
        ],
        test_data_partition=[
                        'AB01_Jimin', 'AB02_Rajiv', 'AB03_Amy',
        ],
        test_data_condition=[
                        #'0mps',
                        '0p2mps', '0p4mps', '0p6mps', '0p8mps', '1p0mps', '1p2mps', '1p4mps', 'transient_15sec', 'transient_30sec',
        ]
    )
    data_handler.save_mean_std(save_dir)

    # Define Loss Function and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparam_config['init_lr'], weight_decay=1e-5)
    # Adjust optimizer to include only the FCNN parameters (freeze the TCN)
    # optimizer = torch.optim.Adam(model.linear.parameters(), lr=hyperparam_config['init_lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    # Initialize Trainer
    trainer = Trainer(device, model, wandb_run, criterion, optimizer, scheduler, data_handler, hyperparam_config, save_dir)

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Finish wandb wandb_run
    wandb_run.finish()

if __name__ == '__main__':
    wandb.login(key="9d7294877630de627f0413ca39aebd4c9a387e50")
    if use_sweep:
        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project= hyperparam_config['wandb_project_name'])
        # Start the sweep agent
        wandb.agent(sweep_id, function=train)
    else:
        # For a single training wandb_run, call train() directly
        train()

