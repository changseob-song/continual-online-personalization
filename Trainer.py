import torch
from tqdm.auto import tqdm
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
import gc

class Trainer:
    def __init__(self, device, model, wandb_run, criterion, optimizer, scheduler, data_handler, config, save_dir):
        
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_handler = data_handler
        self.hyperparam_config = config
        self.save_dir = save_dir
        self.run = wandb_run
        
        # Prepare mean and std tensors
        self.label_mean_tensor = torch.tensor(self.data_handler.label_mean, device=self.device)
        self.label_std_tensor = torch.tensor(self.data_handler.label_std, device=self.device)
        
        # For tracking best validation loss
        self.best_val_loss = float('inf')
        self.patience = 10  # early stopping
        self.patience_counter = 0
        
        # For tracking RMSE
        self.train_rmse_list = []
        self.val_rmse_list = []
        
        # Save model architecture
        model_arch = str(model)
        with open("model_arch.txt", "w") as arch_file:
            arch_file.write(model_arch)
        
        # Log the file as an artifact
        artifact = wandb.Artifact('model_architecture', type='model')
        artifact.add_file('model_arch.txt')
        self.run.log_artifact(artifact)
        
        # Store the transfer_learning flag
        self.transfer_learning = config['transfer_learning']
        
    def train_epoch(self, train_loader):
        self.model.train()
        tloss = 0
        trmse = 0
        tacc = 0
        total_samples = 0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        for i, (input, label) in enumerate(train_loader):
            self.optimizer.zero_grad()
            input = input.to(self.device)
            label = label.to(self.device)
            logits = self.model(input)
            loss = self.criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            tloss += loss.item()

            # Denormalize for RMSE calculation
            preds_denorm = logits * self.label_std_tensor + self.label_mean_tensor
            targets_denorm = label * self.label_std_tensor + self.label_mean_tensor
            preds_percentage = self.polar_to_percentage(preds_denorm)
            targets_percentage = self.polar_to_percentage(targets_denorm)

            # Calculate RMSE
            rmse = self.get_RMSE_gaitcycle(preds_percentage, targets_percentage)
            trmse += rmse

            total_samples += 1

            batch_bar.set_postfix(
                loss="{:.04f}".format(tloss / total_samples),
                rmse="{:.03f}".format(trmse / total_samples),
            )
            batch_bar.update()
            del input, label, logits
            torch.cuda.empty_cache()

        batch_bar.close()
        tloss /= len(train_loader)
        trmse /= len(train_loader)
        return tloss, trmse

    def eval_epoch(self, val_loader):
        self.model.eval()
        vloss = 0
        vrmse = 0
        vacc = 0
        total_samples = 0
        batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                input = input.to(self.device)
                label = label.to(self.device)
                logits = self.model(input)
                loss = self.criterion(logits, label)
                vloss += loss.item()

                # Denormalize for RMSE calculation
                preds_denorm = logits * self.label_std_tensor + self.label_mean_tensor
                targets_denorm = label * self.label_std_tensor + self.label_mean_tensor

                preds_percentage = self.polar_to_percentage(preds_denorm)
                targets_percentage = self.polar_to_percentage(targets_denorm)
                
                # Calculate RMSE
                rmse = self.get_RMSE_gaitcycle(preds_percentage, targets_percentage)
                vrmse += rmse

                total_samples += 1

                batch_bar.set_postfix(
                    loss="{:.04f}".format(vloss / total_samples),
                    rmse="{:.03f}".format(vrmse / total_samples),
                )
                batch_bar.update()
                del input, label, logits
                torch.cuda.empty_cache()

        batch_bar.close()
        vloss /= len(val_loader)
        vrmse /= len(val_loader)
        return vloss, vrmse
        
    def train(self):
        torch.cuda.empty_cache()
        gc.collect()
        num_epochs = self.hyperparam_config['epochs']
        for epoch in range(num_epochs):

            print("\nEpoch {}/{}".format(epoch+1, num_epochs))
            curr_lr = float(self.optimizer.param_groups[0]['lr'])

            train_indices, val_indices = self.data_handler.get_train_val_indices()
            train_loader, val_loader = self.data_handler.create_dataloaders(train_indices, val_indices)

            train_loss, train_rmse = self.train_epoch(train_loader)
            val_loss, val_rmse = self.eval_epoch(val_loader)
            
            self.scheduler.step(val_loss)
            print("\tTrain Loss {:.04f}\tRMSE {:.03f}\tLearning Rate {:.7f}".format(
                train_loss, train_rmse, curr_lr))
            print("\tVal Loss {:.04f}\t\tRMSE {:.03f}".format(
                val_loss, val_rmse))

            # Save RMSE values for plotting
            self.train_rmse_list.append(train_rmse)
            self.val_rmse_list.append(val_rmse)

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{self.hyperparam_config['wandb_session_name']}_epoch_{epoch+1}.pt"))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '.pt'))
                torch.save(self.model.tcn.state_dict(), os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '_tcn.pt'))
                torch.save(self.model.linear.state_dict(), os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '_linear.pt'))
                print("Best model saved at epoch {}".format(epoch + 1))
            else:
                self.patience_counter += 1

            # Plot predictions after each epoch
            test_loader = self.data_handler.create_dataloaders(test_indices=1)

            test_loss, test_rmse = self.eval_epoch(test_loader)
            print("\nTest Loss {:.04f}\tRMSE {:.03f}".format(
                test_loss, test_rmse))


            # Early Stopping
            if self.patience_counter >= self.patience:
                print("Early stopping triggered")
                # Load the best model
                self.model.load_state_dict(torch.load(os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '.pt')))
                break
            
            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_rmse': train_rmse,
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'learning_rate': curr_lr,
                'test_loss': test_loss,
                'test_rmse': test_rmse,
            })
        
    def evaluate(self):
        # Load the best model
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, self.hyperparam_config['wandb_session_name'] + '.pt')))
                
        # Evaluate on Test Data
        test_loader = self.data_handler.create_dataloaders(test_indices=1)
        test_loss, test_rmse = self.eval_epoch(test_loader)
        
        print("\nTest Loss {:.04f}\tRMSE {:.04f}".format(
            test_loss, test_rmse))
        
    def polar_to_percentage(self, polar_coords):
        # Ensure input is a numpy array
        if isinstance(polar_coords, torch.Tensor):
            polar_coords = polar_coords.detach().cpu().numpy()
        angle = np.arctan2(polar_coords[:, 1], polar_coords[:, 0])
        percentage = (angle + 2*np.pi) % (2 * np.pi) / (2 * np.pi) * 100  # Normalize to (0, 100]
        return percentage

    def get_RMSE_gaitcycle(self, preds, targets):

        error_list = []

        for i in range(len(preds)):
            error = preds[i] - targets[i]
            if error < -50:
                error += 100
            elif error > 50:
                error -= 100
            
            error_list.append(error)
        
        rmse = np.sqrt(np.mean(np.array(error_list) ** 2))
        return rmse