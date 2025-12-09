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
import argparse
import copy

seed=0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"seed is {seed}")

def train(model_key, ddt, loss_criterion, inclusion_factors, train_dl, optimizer, val_dl, num_epochs, save_plot_dir = '.', save_model_dir='.', ES_patience=15, lr_scheduler=None, save_fig = False):
    
    early_stopping = EarlyStopping(patience=ES_patience, min_delta=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Training on device:", device)
    
    rss_factor, ot_factor, bt_factor, rp_factor = inclusion_factors
    ddt = ddt.to(device)

    # Track best accuracy *within this specific run*
    best_run_val_acc = 0.0
    best_epoch = -1
    
    neg_pref_avg_rewards = np.zeros(num_epochs)
    neg_pref_std_rewards = np.zeros(num_epochs)
    pos_pref_avg_rewards = np.zeros(num_epochs)
    pos_pref_std_rewards = np.zeros(num_epochs)

    train_losses = []
    train_accuracies = []
    
    val_losses = []
    val_accuracies = []
    best_model = None

    for epoch in range(num_epochs):
        
        print(f"-----------Epoch {epoch}---------------")
        
        acc_counter = 0
        loss_counter = 0
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
            
            # calculate the loss
            final_loss = loss_criterion(loss_tree_traj, pref_label, RSS_factor=rss_factor, OT_factor=ot_factor, BT_factor=bt_factor, RP_factor=rp_factor)
            
            acc_counter += torch.sum((pred_label == pref_label).float())
            loss_counter += final_loss.detach().cpu().numpy()
            
            final_loss.backward()
            optimizer.step()
            
        avg_train_acc = (acc_counter / len(train_dl)) * 100
        train_accuracies.append(avg_train_acc.to("cpu"))
        train_losses.append(loss_counter / len(train_dl))
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        
        neg_pref_avg_reward = 0
        pos_pref_avg_reward = 0
        neg_pref_rewards = []
        pos_pref_rewards = []
        
        # Validation Loop
        with torch.no_grad():
            val_acc_counter = 0
            val_loss_counter = 0
            for val_pref_demo, val_pref_label in val_dl:
                val_pref_label = val_pref_label.to(device)
                val_pref_demo_train = val_pref_demo.view(len(val_pref_demo)*len(val_pref_demo[0]) * len(val_pref_demo[0][0]), 2).float().to(device)
                val_ones = torch.ones((len(val_pref_demo_train), 1)).float().to(device)
                ddt.forward(ddt.root, val_pref_demo_train, val_ones)
                
                loss_tree = ddt.get_loss()
                loss_tree = loss_tree.reshape(len(val_pref_demo), len(val_pref_demo[0]), len(val_pref_demo[0][0]))
                loss_tree_traj = torch.sum(loss_tree, dim=2)
                
                val_pred_label = torch.argmax(loss_tree_traj, dim=1)
                
                val_final_loss = loss_criterion(loss_tree_traj, val_pref_label, RSS_factor=rss_factor, OT_factor=ot_factor, BT_factor=bt_factor, RP_factor=rp_factor)
                
                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())
                val_loss_counter += val_final_loss.detach().cpu().numpy()
                
                ### store rewards for analysis ###
            
                true_neg_pref_reward = loss_tree_traj[0][val_pref_label.item() - 1].detach().cpu().numpy() # get the reward for the true negative preference
                true_pos_pref_reward = loss_tree_traj[0][val_pref_label.item()].detach().cpu().numpy() # get the reward for the true positive preference
                
                neg_pref_avg_reward += true_neg_pref_reward
                pos_pref_avg_reward += true_pos_pref_reward
                
                neg_pref_rewards.append(true_neg_pref_reward) 
                pos_pref_rewards.append(true_pos_pref_reward)
                
                ### ------------------------------- ###

            val_acc_per_epoch = (val_acc_counter / len(val_dl)) * 100
            val_accuracies.append(val_acc_per_epoch.to("cpu"))
            val_losses.append(val_loss_counter / len(val_dl))
            
            ### store average and std rewards ###
        
            neg_pref_avg_reward /= len(val_dl)
            pos_pref_avg_reward /= len(val_dl)
            neg_pref_avg_rewards[epoch] = neg_pref_avg_reward
            pos_pref_avg_rewards[epoch] = pos_pref_avg_reward
            neg_pref_std_rewards[epoch] = np.std(neg_pref_rewards)
            pos_pref_std_rewards[epoch] = np.std(pos_pref_rewards)
            
            ### ------------------------------- ###

            # --- CRITICAL LOGIC: Save Temp Model ---
            # If this epoch is the best for this specific run, save it to a temp file
            if val_acc_per_epoch > best_run_val_acc:
                best_run_val_acc = val_acc_per_epoch
                best_epoch = epoch
                best_model = copy.deepcopy(ddt)
            # -------------------------------------
    
    if save_fig:
        
        # reward plotting
        fig, ax = plt.subplots(figsize = (12, 8))
        
        ax.errorbar(range(len(neg_pref_avg_rewards)), neg_pref_avg_rewards, yerr=neg_pref_std_rewards, label='Negative Preference Reward', fmt='-o')
        ax.errorbar(range(len(pos_pref_avg_rewards)), pos_pref_avg_rewards, yerr=pos_pref_std_rewards, label='Positive Preference Reward', fmt='-o')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Average Reward')
        ax.set_title('Average Rewards for Positive and Negative Preferences Over Epochs')
        ax.legend()
        plt.grid()
        plt_path = os.path.join(save_plot_dir, f"Rewards_Trend_{model_key}.png")
        fig.savefig(plt_path, dpi = 300)
        plt.close(fig)
        
        # loss plotting
        fig, ax = plt.subplots(figsize = (12, 8))
        
        ax.plot(range(len(train_losses)), train_losses, label='Training Loss')
        ax.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Average Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        plt.grid()
        plt_path = os.path.join(save_plot_dir, f"Loss_Trend_{model_key}.png")
        fig.savefig(plt_path, dpi = 300)
        plt.close(fig)
        
        # accuracy plotting
        fig, ax = plt.subplots(figsize = (12, 8))
        
        ax.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
        ax.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_title('Average Accuracy for Positive and Negative Preferences Over Epochs')
        ax.legend()
        plt.grid()
        plt_path = os.path.join(save_plot_dir, f"Accuracy_Trend_{model_key}.png")
        fig.savefig(plt_path, dpi = 300)
        plt.close(fig)
    
    return best_run_val_acc.item(), best_epoch, best_model

if __name__== '__main__':
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model_name', default=None, help="which model do you want? RSS, OT, BT etc")
    parser.add_argument('--soft_routing_argmax', default="hard", type=str, help="Either 'hard' or 'soft'")
    parser.add_argument('--rp_factor', default=0, type=float, help="Reward factor for RP")
    parser.add_argument('--rss_factor', default=0, type=float, help="Reward factor for RSS")
    parser.add_argument('--ot_factor', default=0, type=float, help="Reward factor for OT")
    parser.add_argument('--bt_factor', default=0, type=float, help="Reward factor for BT")
    parser.add_argument('--learning_rate', default=1e-3, type=float, help="Learning rate for the optimizer")
    parser.add_argument('--num_epochs', default=10, type=int, help="Number of training epochs")
    parser.add_argument('--depth', default=2, type=int, help="Depth of the DDT")
    parser.add_argument('--save_figures', default=True, type=bool, help="Whether to save figures")
    
    args = parser.parse_args()
    
    if args.model_name is None:
        print("Model name must be specified. Use --model_name to specify the model.")
        exit(1)
        
    if args.rp_factor == 0 and args.rss_factor == 0 and args.ot_factor == 0 and args.bt_factor == 0:
        print("At least one reward factor must be non-zero.")
        exit(1)

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
    save_figures = args.save_figures
    
    # Hyperparameters
    loss_criterion = BT_OT_RSS_Loss
    model_key = args.model_name
    num_epochs = args.num_epochs
    lr = args.learning_rate
    reward_strat = args.soft_routing_argmax
    rss = args.rss_factor
    ot = args.ot_factor
    bt = args.bt_factor
    rp = args.rp_factor
    factors = (rss, ot, bt, rp)
    depth = args.depth
    
    # print hyperparameters
    print(f"Model Name: {model_key}")
    print(f"Depth: {depth}")
    print(f"Reward Strategy: {reward_strat}")
    print(f"RSS Factor: {rss}")
    print(f"OT Factor: {ot}")
    print(f"BT Factor: {bt}")
    print(f"RP Factor: {rp}")
    print(f"Learning Rate: {lr}")
    print(f"Number of Epochs: {num_epochs}")
    
    # Constant parameters
    class_reward_vector = [0, 0.25]
    nb_classes = len(class_reward_vector)
    weight_decay=0.0

    # Setup Paths
    current_directory = os.getcwd() + '/logic/Final_Models_50_epochs/'
    base_save_dir = os.path.join(current_directory, 'DDT')
    save_model_dir = os.path.join(base_save_dir, 'saved_models')
    save_config_dir = os.path.join(base_save_dir, 'configs')
    save_plot_dir = os.path.join(base_save_dir, f'plots/{model_key}')
    
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_config_dir, exist_ok=True)
    os.makedirs(save_plot_dir, exist_ok=True)

    # Init Model
    tree = SoftDecisionTree(depth, nb_classes, input_dim, class_reward_vector, seed=seed, reward_strategy=reward_strat)
    optimizer = optim.Adam(tree.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\n--- Running ---")

    # --- RUN TRAINING ---
    # The function saves a file named "TEMP_{Exp_name}.pth" when it finds a local best
    val_acc, best_epoch, best_model = train(model_key, tree, loss_criterion, factors, train_dl, optimizer, val_dl, num_epochs=num_epochs, 
                    save_plot_dir=save_plot_dir, save_model_dir=save_model_dir, save_fig=save_figures)
        
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
        'best_epoch': best_epoch,
        'num_train_prefs': num_train_prefs,
    }

    # 1. Rename the Temp Model to Final Model
    final_model_name = f"{model_key}.pth" # e.g. BEST_RSS_hard.pth
    final_model_path = os.path.join(save_model_dir, final_model_name)
    
    # save the model
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    temp_path = os.path.join(save_model_dir, f"{model_key}_best.pth")
    torch.save(best_model.state_dict(), temp_path)
    best_temp_path = temp_path

    # 2. Save the Config immediately
    config_filename = f"{model_key}_config.yaml"
    # Create specific subfolder if desired, or dump in main config dir
    config_path = os.path.join(save_config_dir, config_filename)
    
    with open(config_path, "w") as f:
        yaml.dump(final_config, f)
    
    print(f"Saved Config: {config_filename}")
    print(f"Saved Model: {final_model_name}")