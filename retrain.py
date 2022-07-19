import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import math 
import optuna
from utils.train import (
    seed_all,
    set_model,
    get_calculated_means_stds_trainval, 
    get_patch_lists, 
    get_loaders, 
    train_epoch,
    save_checkpoint
)
from utils.parser import create_train_parser

def retrain_best_trial(args):
    seed_all(seed=args.seed)
    architecture:str = args.architecture
    encoder_name:str = args.encoder_name
    
    root_path: str = args.root_path
    if args.db_name =="":
        db_name:str = f"retrain_{architecture}_{encoder_name}"
    else:
        db_name = args.db_name
    print(f"loaded db {db_name}")
    # Parameters
    max_epochs = 100
    es_patience = 5
    loss_total = 1
    epochs_no_improve = 0
    # NEED TO CHANGE THIS LINE OF CODE TO RE-TRAIN DIFFERENT MODELS
    study_storage = f"sqlite:///{root_path}/results/studies/{architecture}/save_{architecture}_{encoder_name}_dil0_bilin1_pre1.db"
    studies = optuna.study.get_all_study_summaries(storage=study_storage)
    loaded_study = optuna.load_study(study_name=studies[0].study_name, storage=study_storage)
    trial = loaded_study.best_trial
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Study: {studies[0].study_name} from {db_name}")
    print(f"Best Trial:{trial.number}")
    print(trial)
    
    model_path = Path(f'{root_path}/models/')
    model_path.mkdir(parents=True, exist_ok=True)
    result_path = Path(f'{root_path}/results/')
    result_path.mkdir(parents=True, exist_ok=True)

    # extract hyperparameters, feature extractor and architecture from best trial
    lr = trial.params["lr"]
    lr_factor = trial.params["lr_factor"]
    batch_size=trial.user_attrs["batch_size"]
    lr_scheduler_patience = trial.user_attrs["lr_scheduler_patience"]
    architecture = trial.user_attrs["architecture"]
    encoder_name = trial.user_attrs["encoder_name"]
    pretrained = trial.user_attrs["pretrained"]
    b_bilinear = trial.user_attrs["b_bilinear"]
    replace_stride_with_dilation = trial.user_attrs["replace_stride"]
    data_path = Path(root_path) / "data" 

    train_img_dir, train_msk_dir = get_patch_lists(
    data_path=data_path, 
    subset="trainval")

    valid_img_dir, valid_msk_dir = get_patch_lists(
    data_path=data_path, 
    subset="test")


    model_save_str = f"model_{architecture}_{encoder_name}_dil{int(replace_stride_with_dilation)}_bilin{int(b_bilinear)}_retrained.pt"
    model_save_path = Path(root_path) / "models" / model_save_str
    model = set_model(architecture=architecture, encoder_name=encoder_name, pretrained=pretrained, b_bilinear=b_bilinear, replace_stride_with_dilation=replace_stride_with_dilation, num_classes=3).to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor*0.1, min_lr=1e-6, patience=lr_scheduler_patience)
    means, stds = get_calculated_means_stds_trainval()       

    train_loader, _ = get_loaders(
            train_img_dir = train_img_dir,
            train_msk_dir = train_msk_dir,
            valid_img_dir = valid_img_dir, 
            valid_msk_dir = valid_msk_dir,
            mean = means,
            std = stds,
            batch_size = batch_size,
            num_workers = 4,
            pin_memory = True,
        )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(max_epochs):
        train_loss = train_epoch(
            train_loader, 
            model, 
            optimizer, 
            loss_fn, 
            scaler, 
            cur_epoch=epoch
            )
        checkpoint = {
            "state_dict": model.state_dict(),
        }
        scheduler.step(train_loss)
        if train_loss < loss_total:
            loss_total = train_loss
            print(f"Saving checkpoint in epoch {epoch}...")
            save_checkpoint(checkpoint, filename=f"{str(model_save_path)}")
        else:
            epochs_no_improve+=1
        # sometimes it can happen, that test_loss is nan --> cannot save nan to database, so we need to change it
        if math.isnan(train_loss):
            train_loss = 99999
        if epochs_no_improve >= es_patience:
            print(f"Early Stopping on epoch {epoch}")
            break
        print(f"Loss on Train set: {train_loss}")
    return train_loss



if __name__ == "__main__":
    args = create_train_parser()
    retrain_best_trial(args)
