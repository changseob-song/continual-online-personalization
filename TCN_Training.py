# ICORR_TCN_Training.py
import torch
import os
import wandb

from TCN_Header_Model import TCNModel
from TCN_Header_Dataloader import DataHandler
from TCN_Header_Trainer import Trainer


use_sweep = False  # Set to True to run sweep, False for single run
# Define the sweep configuration outside the main function
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'RMSE',
        'goal': 'minimize'
    },
    'parameters': {
        'dataset_proportion': {'values': [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2]},
    }
}

# Base hyperparameters
hyperparam_config = {
    'wandb_project_name': 'Biotorque_initial',
    'wandb_session_name': 'bilateral_to_unilateral_wo_pelvis_test',
    'input_size': 14, # 12 for IMU (right, left), 2 for hip angle and velocity
    'output_size': 1, # 1 for right hip torque
    'architecture': 'TCN',
    
    'transfer_learning': False,
    'dataset_proportion': 1, # dataset proportion for training
    
    'epochs': 30,
    'batch_size': 32,
    'init_lr': 5e-4,
    'dropout': 0.15,
    'validation_split': 0.1,
    'window_size': 95,
    'number_of_layers': 2,
    'num_channels': [50, 50, 50, 50, 50],
    'kernel_size': 5,
    'dilations': [1, 2, 4, 8, 16],
    'number_of_workers': 10
}

def train():

    # Initialize wandb run with hyperparameters
    wandb_run = wandb.init(config=hyperparam_config, project=hyperparam_config['wandb_project_name'], name=hyperparam_config['wandb_session_name'])

    # Access wandb.config after initializing wandb
    config = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Update hyperparameters with wandb config (useful if overridden)
    hyperparam_config.update(dict(config))

    # Create directory for results & plots
    save_dir = '/home/metamobility3/Changseob/biotorque/in-lab_version/training_result'
    save_sub_dir = hyperparam_config['wandb_session_name']  # 1. AB, 2. SK, 3. AB+SK
    save_dir = os.path.join(save_dir, save_sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Model Initialization
    model = TCNModel(hyperparam_config).to(device)
    
    # Load pretrained model if transfer learning is enabled
    if hyperparam_config['transfer_learning']:
        pretrained_model_path = '/home/metamobility/Changseob/Initial_Project/ICORR_2025_TCN/results/trained_model/AB'
        model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'AB_model.pt'), map_location=device))
        print("\nPretrained model loaded: ", pretrained_model_path)
        # #Freeze the TCN part of the model
        # for param in model.tcn.parameters():
        #     param.requires_grad = False
    else:
        pretrained_model_path = None

    # Initialize DataHandler
    data_root = '/home/metamobility3/Changseob/biotorque/in-lab_version/biotorque_ten_subjects'
    data_handler = DataHandler(data_root, hyperparam_config, pretrained_model_path)
    data_handler.load_data(
        train_data_partition=[
                        'AB01_Jimin',
                        'AB02_Rajiv',
                        'AB03_Amy',
                        'AB04_Changseob',
                        'AB05_Maria',
                        # 'AB06_Vaidehi',
                        'AB07_Leo',
                        'AB08_Adrian',
                        'AB09_Crystal',
        ],
        test_data_partition=[
                        # 'AB01_Jimin',
                        # 'AB02_Rajiv',
                        # 'AB03_Amy',
                        # 'AB04_Changseob',
                        # 'AB05_Maria',
                        'AB06_Vaidehi',
                        # 'AB07_Leo',
                        # 'AB08_Adrian',
                        # 'AB09_Crystal',
                        # "AB06_Vaidehi_test",
                        # 'PT01_Changseob'
        ],
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
    
    if use_sweep:
        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project= hyperparam_config['wandb_project_name'])
        # Start the sweep agent
        wandb.agent(sweep_id, function=train)
    else:
        # For a single training wandb_run, call train() directly
        train()

