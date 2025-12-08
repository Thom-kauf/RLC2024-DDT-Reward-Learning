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
from train_model import train

seed=0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"seed is {seed}")

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
        save_plot_dir = os.path.join(base_save_dir, 'plots')
        
        os.makedirs(save_model_dir, exist_ok=True)
        os.makedirs(save_config_dir, exist_ok=True)
        os.makedirs(save_plot_dir, exist_ok=True)

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
                        save_plot_dir=save_plot_dir, save_model_dir=save_model_dir, model_key=model_key, ES_patience=10)
        
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