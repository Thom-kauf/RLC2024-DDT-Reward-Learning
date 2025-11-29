def train(ddt,train_dl, optimizer,val_dl, num_epochs,save_model_dir='.',exp_no=0,ES_patience=15,lr_scheduler=None):

    early_stopping = EarlyStopping(patience=ES_patience, min_delta=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # loss_criterion = nn.CrossEntropyLoss()

    # loss_criterion = Richardson_Srikumar_Sabhahwal_Loss
    # loss_criterion = RSS_OT_Loss

    loss_criterion = BT_RSS_Loss

    # loss_criterion = Richardson_Srikumar_Sabhahwal_Loss

    # loss_criterion = One_True_Loss

    ddt = ddt.to(device)

    global_step = 0

    for epoch in range(num_epochs):
        acc_counter = 0
        losses = []

        if lr_scheduler!=None:
            print(f"-----------Epoch{epoch} and lr is {lr_scheduler.get_last_lr()}  ---------------")
        else:
            print(f"-----------Epoch{epoch}---------------")
        for pref_demo, pref_label in train_dl:
            optimizer.zero_grad()
            pref_label = pref_label.to(device)
            pref_demo_train = pref_demo.view(len(pref_demo)*len(pref_demo[0])*len(pref_demo[0][0]),2).float().to(device)
            ones = torch.ones((len(pref_demo_train), 1)).float().to(device)
            ddt.forward(ddt.root, pref_demo_train, ones)
            loss_tree = ddt.get_loss()
            loss_tree = loss_tree.reshape(len(pref_demo),len(pref_demo[0]), len(pref_demo[0][0]))
            loss_tree_traj = torch.sum(loss_tree, dim=2)


            pred_label = torch.argmax(loss_tree_traj, dim=1)
            # print(f"pred label is {pred_label} and pref label is {pref_label}")
            acc_counter += torch.sum((pred_label == pref_label).float())
            final_loss = loss_criterion(loss_tree_traj, pref_label)#, RSS_factor=1e3, BT_factor=1)

            # print(f"Pos reward from r_theta is {loss_tree_traj[0, pref_label.item()]} and neg reward is {loss_tree_traj[0, pref_label.item()] - 1}")
            # print(f"final loss is {final_loss.item()}")

            losses.append(final_loss.detach().cpu().numpy())

            writer.add_scalar('Training Loss per step', final_loss.detach().cpu().numpy(), global_step)
            global_step += 1



            final_loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        training_loss_per_epoch = np.mean(losses)
        print("Training Loss per epoch", training_loss_per_epoch)
        training_acc_per_epoch = acc_counter / (len(train_dl)*len(pref_demo)) * 100
        print(" Training Accuracy per epoch", training_acc_per_epoch)
        writer.add_scalar('Training Loss per epoch', training_loss_per_epoch, epoch)
        writer.add_scalar(' Training Accuracy per epoch', training_acc_per_epoch, epoch)

        with torch.no_grad():
            val_acc_counter = 0
            val_losses = []
            for val_pref_demo, val_pref_label in val_dl:

                val_pref_label = val_pref_label.to(device)
                val_pref_demo_train = val_pref_demo.view(len(val_pref_demo)*len(val_pref_demo[0]) * len(val_pref_demo[0][0]), 2).float().to(device)
                val_ones = torch.ones((len(val_pref_demo_train), 1)).float().to(device)
                ddt.forward(ddt.root, val_pref_demo_train, val_ones)
                val_loss_tree = ddt.get_loss()
                val_loss_tree = val_loss_tree.reshape(len(val_pref_demo), len(val_pref_demo[0]), len(val_pref_demo[0][0]))
                val_loss_tree_traj = torch.sum(val_loss_tree, dim=2)

                val_pred_label = torch.argmax(val_loss_tree_traj, dim=1)
                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())

                val_final_loss = loss_criterion(val_loss_tree_traj, val_pref_label)
                
                val_losses.append(val_final_loss.detach().cpu().numpy())


            val_loss_per_epoch = np.mean(val_losses)
            print("Val Loss per epoch", val_loss_per_epoch)
            val_acc_per_epoch = val_acc_counter / (len(val_dl)*len(val_pref_demo)) * 100
            print("VAL Accuracy per epoch", val_acc_per_epoch)
            writer.add_scalar('Val Loss per epoch', val_loss_per_epoch, epoch)
            writer.add_scalar('Val Accuracy per epoch', val_acc_per_epoch, epoch)
            '''use this for ReduceLRonPlateau- NOT USING IT RIGHT NOW'''
            # if lr_scheduler is not None:
            #     scheduler.step(val_loss_per_epoch)
            early_stopping(val_loss_per_epoch)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                torch.save(ddt, save_model_dir + exp_no + "_" + str(epoch))
                break
    if early_stopping.early_stop:
        pass
    elif not early_stopping.early_stop:
        torch.save(ddt, save_model_dir + exp_no + "_" + str(num_epochs))
        print(f"no of epochs are {num_epochs}")

if __name__== '__main__':

    '''prep data'''
    num_prefs=2200
    traj_snippet_len=20
    pref_dataset_path='Pref_Dataset_num_prefs_'+str(num_prefs)+'_traj_snippet_len_'+str(traj_snippet_len)
    pref_dataset=torch.load(pref_dataset_path)
    pref_demos=pref_dataset['pref_demos']
    pref_labels=pref_dataset['pref_labels']
    assert len(pref_demos) == len(pref_labels) == num_prefs
    num_train_prefs=2000

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

    save_config=True
    input_dim = 1 * 2



    # to tune
    depth = 2
    class_reward_vector = [0, 1]#0.01]
    nb_classes = len(class_reward_vector)
    tree = SoftDecisionTree(depth, nb_classes, input_dim, class_reward_vector, seed=seed)

    # will need to tune this
    lr=0.001
    weight_decay=0.000

    optimizer = optim.Adam(tree.parameters(), lr=lr, weight_decay=weight_decay)
    Exp_name = 'CP-DDT-1'
    current_directory = os.getcwd()
    save_model_dir = current_directory +'/Reward_Models/DDT/saved_models/'
    tensorboard_path = current_directory +'/Reward_Models/DDT/TB/' + Exp_name

    writer = SummaryWriter(tensorboard_path)
    if not os.path.exists(save_model_dir):
        print(' Creating Project : ' + save_model_dir)
        os.makedirs(save_model_dir)

    if save_config:
        config=dict()
        config['seed'] = seed
        config['input_dim'] = input_dim
        config['depth'] = depth
        config['class_reward_vector'] = class_reward_vector
        config['lr'] = lr
        config['weight_decay'] = weight_decay
        config[' num_train_prefs'] = num_train_prefs
        config['train_dl_len']=train_dl_len
        config['val_dl_len']=val_dl_len

        save_config_dir = current_directory +'/Reward_Models/DDT/configs/'
        if not os.path.exists(save_config_dir):
            print('Creating Project : ' + save_config_dir)
            os.makedirs(save_config_dir)
        path = save_config_dir + Exp_name + "_config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

    train(tree, train_dl, optimizer, val_dl, num_epochs=20, save_model_dir=save_model_dir, exp_no=Exp_name,
          ES_patience=10, lr_scheduler=None)
