import torch
import random
import torch.nn as nn
import numpy as np
import os
import shutil 
import yaml
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from Utils import EarlyStopping
from Reward_DDT import SoftDecisionTree
from logic.Reward_Losses import * 
from matplotlib import pyplot as plt
seed=0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"seed is {seed}")

def train(ddt, loss_criterion, inclusion_factors, train_dl, optimizer, val_dl, num_epochs, base_model_dir = '.',save_model_dir='.', exp_no='0', ES_patience=15, lr_scheduler=None, save_fig = False):
    
    early_stopping = EarlyStopping(patience=ES_patience, min_delta=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    rss_factor, ot_factor, bt_factor, rp_factor = inclusion_factors
    ddt = ddt.to(device)

    # Track best accuracy *within this specific run*
    best_run_val_acc = 0.0
    best_epoch = -1
    
    neg_pref_avg_rewards = np.zeros(num_epochs)
    neg_pref_std_rewards = np.zeros(num_epochs)
    pos_pref_avg_rewards = np.zeros(num_epochs)
    pos_pref_std_rewards = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        acc_counter = 0
        losses = []
        
        # print(f"-----------Epoch{epoch}---------------")
        
        # Training loop
        for pref_demo, pref_label in train_dl:
            optimizer.zero_grad()
            pref_label = pref_label.to(device)
            pref_demo_train = pref_demo.view(len(pref_demo)*len(pref_demo[0])*len(pref_demo[0][0]),2).float().to(device)
            ones = torch.ones((len(pref_demo_train), 1)).float().to(device)
            
            ddt.forward(ddt.root, pref_demo_train, ones)
            
            # the loss tree is the reward for each state in each trajectory
            loss_tree = ddt.get_loss()
            loss_tree = loss_tree.reshape(len(pref_demo),len(pref_demo[0]), len(pref_demo[0][0]))
            
            # for each state in the trajectory, sum over the rewards to get the trajectory reward
            loss_tree_traj = torch.sum(loss_tree, dim=2)
            
            # gets the preferred trajectory index (the trajectory with the highest reward)
            pred_label = torch.argmax(loss_tree_traj, dim=1)
            
            acc_counter += torch.sum((pred_label == pref_label).float())
            
            # calculate the loss
            final_loss = loss_criterion(loss_tree_traj, pref_label, RSS_factor=rss_factor, OT_factor=ot_factor, BT_factor=bt_factor, RP_factor=rp_factor)
            losses.append(final_loss.detach().cpu().numpy())
            
            final_loss.backward()
            optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        training_loss_per_epoch = np.mean(losses)
        # print("Training Loss per epoch", training_loss_per_epoch)
        
        neg_pref_avg_reward = 0
        pos_pref_avg_reward = 0
        neg_pref_rewards = []
        pos_pref_rewards = []
        
        # Validation Loop
        with torch.no_grad():
            val_acc_counter = 0
            val_losses = []
            for val_pref_demo, val_pref_label in val_dl:
                val_pref_label = val_pref_label.to(device)
                val_pref_demo_train = val_pref_demo.view(len(val_pref_demo)*len(val_pref_demo[0]) * len(val_pref_demo[0][0]), 2).float().to(device)
                val_ones = torch.ones((len(val_pref_demo_train), 1)).float().to(device)
                ddt.forward(ddt.root, val_pref_demo_train, val_ones)
                
                loss_tree = ddt.get_loss()
                loss_tree = loss_tree.reshape(len(val_pref_demo), len(val_pref_demo[0]), len(val_pref_demo[0][0]))
                loss_tree_traj = torch.sum(loss_tree, dim=2)
                
                val_pred_label = torch.argmax(loss_tree_traj, dim=1)
                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())
                val_final_loss = loss_criterion(loss_tree_traj, val_pref_label, RSS_factor=rss_factor, OT_factor=ot_factor, BT_factor=bt_factor, RP_factor=rp_factor)
                val_losses.append(val_final_loss.detach().cpu().numpy())
                
                ### store rewards for analysis ###
            
                true_neg_pref_reward = loss_tree_traj[0][val_pref_label.item() - 1].detach().cpu().numpy() # get the reward for the true negative preference
                true_pos_pref_reward = loss_tree_traj[0][val_pref_label.item()].detach().cpu().numpy() # get the reward for the true positive preference
                
                neg_pref_avg_reward += true_neg_pref_reward
                pos_pref_avg_reward += true_pos_pref_reward
                
                neg_pref_rewards.append(true_neg_pref_reward) 
                pos_pref_rewards.append(true_pos_pref_reward)
                
                ### ------------------------------- ###

            val_loss_per_epoch = np.mean(val_losses)
            val_acc_per_epoch = val_acc_counter / (len(val_dl)*len(val_pref_demo)) * 100
            
            ### store average and std rewards ###
        
            neg_pref_avg_reward /= len(val_dl)
            pos_pref_avg_reward /= len(val_dl)
            neg_pref_avg_rewards[epoch] = neg_pref_avg_reward
            pos_pref_avg_rewards[epoch] = pos_pref_avg_reward
            neg_pref_std_rewards[epoch] = np.std(neg_pref_rewards)
            pos_pref_std_rewards[epoch] = np.std(pos_pref_rewards)
            
            ### ------------------------------- ###
            
            # print("Val Loss per epoch", val_loss_per_epoch)
            # print("VAL Accuracy per epoch", val_acc_per_epoch)

            # --- CRITICAL LOGIC: Save Temp Model ---
            # If this epoch is the best for this specific run, save it to a temp file
            if val_acc_per_epoch > best_run_val_acc:
                best_run_val_acc = val_acc_per_epoch
                best_epoch = epoch
                # Ensure directory exists
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                
                temp_path = os.path.join(save_model_dir, f"TEMP_{exp_no}.pth")
                torch.save(ddt.state_dict(), temp_path)
            # -------------------------------------

            early_stopping(val_loss_per_epoch)
            if early_stopping.early_stop:
                print("Early stopping at epoch:", epoch)
                break
    
    if save_fig:
        fig, ax = plt.subplots(figsize = (12, 8))
        
        ax.errorbar(range(len(neg_pref_avg_rewards)), neg_pref_avg_rewards, yerr=neg_pref_std_rewards, label='Negative Preference Reward', fmt='-o')
        ax.errorbar(range(len(pos_pref_avg_rewards)), pos_pref_avg_rewards, yerr=pos_pref_std_rewards, label='Positive Preference Reward', fmt='-o')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Average Reward')
        ax.set_title('Average Rewards for Positive and Negative Preferences Over Epochs')
        ax.legend()
        plt.grid()
        plt_path = os.path.join(base_model_dir, f"plots/Rewards_Trend_{exp_no}.png")
        plt.savefig(plt_path, dpi = 300)
        plt.close(fig)
    
    return best_run_val_acc.item(), best_epoch


if __name__== '__main__':

    # --- DATA PREP (RESTORED) ---
    num_prefs= 2200
    traj_snippet_len=20
    pref_dataset_path='Pref_Dataset_num_prefs_'+str(num_prefs)+'_traj_snippet_len_'+str(traj_snippet_len)
    
    pref_dataset=torch.load(pref_dataset_path)
    pref_demos=pref_dataset['pref_demos']
    pref_labels=pref_dataset['pref_labels']
    assert len(pref_demos) == len(pref_labels) == num_prefs
    
    # Your slice indices
    num_train_prefs = 2000
    
    train_pref_demos=pref_demos[:num_train_prefs]
    train_pref_labels=pref_labels[:num_train_prefs]

    val_pref_demos=pref_demos[num_train_prefs:]
    val_pref_labels=pref_labels[num_train_prefs:]

    train_dataset = TensorDataset(torch.stack(train_pref_demos),torch.tensor(train_pref_labels))
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=False)

    val_datset = TensorDataset(torch.stack(val_pref_demos),torch.tensor(val_pref_labels))
    val_dl = DataLoader(val_datset, batch_size=1, shuffle=False)
    
    val_dl_len=len(val_dl)
    train_dl_len=len(train_dl)


    input_dim = 1 * 2

    # Hyperparameters
    lrs = [1e-4, 1e-3, 1e-2] 
    inclusion_factors = {
        'RSS_factor': [0], 
        'BT_factor': [0], 
        'OT_factor': [0],
        'RP_factor': [0, 1e0, 1e2]
    }
    
    reward_strategies = ["hard"] 

    hyperparameters_grid = []
    for RSS_factor in inclusion_factors['RSS_factor']:
        for BT_factor in inclusion_factors['BT_factor']:
            for OT_factor in inclusion_factors['OT_factor']:
                for RP_factor in inclusion_factors['RP_factor']:
                    for reward_strategy in reward_strategies:
                        for lr in lrs:
                            hyperparameters_grid.append({
                                'RSS_factor': RSS_factor, 'BT_factor': BT_factor, 'OT_factor': OT_factor, 'RP_factor': RP_factor,
                                'reward_strategy': reward_strategy, 'lr': lr
                            })
            
    print(f"Total hyperparameter combinations to try: {len(hyperparameters_grid)}")
    
    # Constant parameters
    depth = 2
    class_reward_vector = [0, 0.25]
    nb_classes = len(class_reward_vector)
    weight_decay=0.0
    num_epochs = 5

    # --- STEP 1: INITIALIZE DICTIONARY CORRECTLY (14 KEYS) ---
    # We create unique keys for every Loss Type + Reward Strategy combo
    best_acc_dict = {}

    print("Tracking the following Model Types:", list(best_acc_dict.keys()))

    # --- MAIN LOOP ---
    loss_criterion = BT_OT_RSS_Loss
    for hyperparameters in hyperparameters_grid:
        
        lr = hyperparameters['lr']
        reward_strat = hyperparameters['reward_strategy']
        rss, ot, bt, rp = (hyperparameters['RSS_factor'], hyperparameters['OT_factor'], hyperparameters['BT_factor'], hyperparameters['RP_factor'])
        factors = (rss, ot, bt, rp)

        if rss == 0 and ot == 0 and bt == 0 and rp == 0:
            print("Skipping all-zero factors")
            continue

        # Setup Paths
        current_directory = os.getcwd() + '/logic/'
        base_save_dir = os.path.join(current_directory, 'Reward_Models_Shifted', 'DDT')
        save_model_dir = os.path.join(base_save_dir, 'saved_models')
        save_config_dir = os.path.join(base_save_dir, 'configs')
        
        os.makedirs(save_model_dir, exist_ok=True)
        os.makedirs(save_config_dir, exist_ok=True)

        # Unique ID for this specific run
        Exp_name = f"Strat_{reward_strat}_RSS_{rss:.0e}_OT_{ot:.0e}_BT_{bt:.0e}_RP_{rp:.0e}_LR_{lr:.0e}_TrainPrefs_{num_train_prefs}"
        tensorboard_path = os.path.join(base_save_dir, 'TB', Exp_name)
        writer = SummaryWriter(tensorboard_path)

        # Init Model
        tree = SoftDecisionTree(depth, nb_classes, input_dim, class_reward_vector, seed=seed, reward_strategy=reward_strat)
        optimizer = optim.Adam(tree.parameters(), lr=lr, weight_decay=weight_decay)

        print(f"\n--- Running: {Exp_name} ---")

        # --- RUN TRAINING ---
        # The function saves a file named "TEMP_{Exp_name}.pth" when it finds a local best
        val_acc, best_epoch = train(tree, loss_criterion, factors, train_dl, optimizer, val_dl, num_epochs=num_epochs, 
                        base_model_dir=base_save_dir,save_model_dir=save_model_dir, exp_no=Exp_name, ES_patience=10)
        
        # --- STEP 2: DETERMINE BASE MODEL TYPE ---
        base_type = ""
        if rss > 0: base_type += "RSS"
        if ot > 0:
            if base_type != "": base_type += "_"
            base_type += "OT"
        if bt > 0:
            if base_type != "": base_type += "_"
            base_type += "BT"
        if rp > 0:
            if base_type != "": base_type += "_"
            base_type += "RP"
        
        # --- STEP 3: COMBINE WITH STRATEGY FOR KEY ---
        # This creates keys like "RSS_hard" and "RSS_soft"
        # This ensures we save a best model for hard AND a best model for soft
        model_key = f"{base_type}_{reward_strat}" 

        # --- STEP 4: COMPARE AND FINALIZE SAVE ---
        if val_acc > best_acc_dict.get(model_key, 0.0):
            best_acc_dict[model_key] = val_acc
            
            # Prepare Config Data
            final_config = {
                'seed': seed,
                'input_dim': input_dim,
                'depth': depth,
                'class_reward_vector': class_reward_vector,
                'lr': lr,
                'weight_decay': weight_decay,
                'RSS_factor': rss,
                'OT_factor': ot,
                'BT_factor': bt,
                'RP_factor': rp,
                'reward_strategy': reward_strat,
                'best_val_acc': val_acc,
                'source_exp_name': Exp_name,
                'best_epoch': best_epoch
            }

            # 1. Rename the Temp Model to Final Model
            temp_model_path = os.path.join(save_model_dir, f"TEMP_{Exp_name}.pth")
            final_model_name = f"BEST_{model_key}_{num_train_prefs}_train_prefs.pth" # e.g. BEST_RSS_hard.pth
            final_model_path = os.path.join(save_model_dir, final_model_name)
            
            # Use move/rename
            if os.path.exists(temp_model_path):
                # If a previous best file exists, this will overwrite it, which is what we want
                shutil.move(temp_model_path, final_model_path)
                final_config['model_path'] = final_model_path
            else:
                print(f"Warning: Temp model file not found at {temp_model_path}")

            # 2. Save the Config immediately
            config_filename = f"BEST_{model_key}_{num_train_prefs}_train_prefs_config.yaml"
            # Create specific subfolder if desired, or dump in main config dir
            config_path = os.path.join(save_config_dir, config_filename)
            
            with open(config_path, "w") as f:
                yaml.dump(final_config, f)
            
            print(f"Saved Config: {config_filename}")
            print(f"Saved Model: {final_model_name}")
        
        else:
            # Clean up temp file if it wasn't the global best for this category
            temp_model_path = os.path.join(save_model_dir, f"TEMP_{Exp_name}.pth")
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    print("\nFinal Best Accuracies:")
    print(best_acc_dict)