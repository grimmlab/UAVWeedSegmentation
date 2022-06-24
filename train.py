import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import math 
import kornia

from utils.train import (
    seed_all,
    set_study,
    set_model,
    get_calculated_means_stds_per_fold, 
    get_patch_lists, 
    get_loaders, 
    save_checkpoint,
    train_epoch,
    validate_epoch,
)
from utils.parser import create_train_parser


def objective(trial):
    epochs_no_improve:int = 0
    kfold = KFold(n_splits=num_folds, shuffle=False)
    loss_total = np.ones(num_folds)*99999
    epochs = np.ones(num_folds)*0
    img_list, msk_list = get_patch_lists(
    data_path=data_path, 
    subset="trainval")
    for fold, (train_ids, val_ids) in enumerate(kfold.split(img_list)):
        train_img_dir = [img_list[i] for i in train_ids]
        train_msk_dir = [msk_list[i] for i in train_ids]
        valid_img_dir = [img_list[i] for i in val_ids]
        valid_msk_dir = [msk_list[i] for i in val_ids]
        epochs_no_improve = 0

        model = set_model(architecture=architecture, encoder_name=encoder_name, pretrained=pretrained, b_bilinear=b_bilinear, replace_stride_with_dilation=replace_stride_with_dilation, num_classes=3).to(device=device)
        
        loss_fn = kornia.losses.DiceLoss()
        lr = trial.suggest_loguniform("lr", lr_ranges[0], lr_ranges[1])
        print(f"suggested LR: {lr}")
        reduce_factor = trial.suggest_int("lr_factor", int(lr_factor_ranges[0]*10), int(lr_factor_ranges[1]*10), step=int(lr_factor_ranges[2]*10))
        reduce_factor = reduce_factor*0.1
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=reduce_factor, min_lr=lr_ranges[0], patience=lr_scheduler_patience)
        means, stds = get_calculated_means_stds_per_fold(fold)
        train_loader, valid_loader = get_loaders(
            train_img_dir = train_img_dir,
            train_msk_dir = train_msk_dir,
            valid_img_dir = valid_img_dir, 
            valid_msk_dir = valid_msk_dir,
            mean = means,
            std = stds,
            batch_size = args.batch_size,
            num_workers = num_workers,
            pin_memory = False,
        )
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(max_epochs):
            train_loss = train_epoch(
                train_loader, 
                model, 
                optimizer, 
                loss_fn, 
                scaler, 
                cur_epoch=epoch,
                trial_number=trial.number,
                fold=fold,
                )
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            
            valid_loss = validate_epoch(
                valid_loader, 
                model, 
                cur_epoch=epoch, 
                trial_number=trial.number,
                fold=fold,
                )
            scheduler.step(valid_loss)
            
            if valid_loss < loss_total[fold]:
                loss_total[fold] = valid_loss
                if b_save_checkpoint:
                    save_checkpoint(checkpoint, filename=f"{str(model_path)}/{architecture}_{encoder_name}_dil{int(replace_stride_with_dilation)}_bilin{int(b_bilinear)}_pre{int(pretrained)}.pth.tar")
            else:
                epochs_no_improve+=1
            # sometimes it can happen, that valid_loss is nan --> cannot save nan to database, so we need to change it
            if math.isnan(valid_loss):
                valid_loss = 99999
            
            if epochs_no_improve >= es_patience:
                print(f"Early Stopping on epoch {epoch}")
                epochs[fold]=epoch
                break

    trial.set_user_attr('Valid loss per fold', list(loss_total))
    trial.set_user_attr('root path', root_path)
    trial.set_user_attr('architecture', architecture)
    trial.set_user_attr('encoder_name', encoder_name)
    trial.set_user_attr('batch_size', batch_size)
    trial.set_user_attr('b_bilinear', b_bilinear)
    trial.set_user_attr('pretrained', pretrained)
    trial.set_user_attr('replace_stride', replace_stride_with_dilation)
    trial.set_user_attr('final_epoch', list(epochs))
    trial.set_user_attr('lr_scheduler_patience', lr_scheduler_patience)
    print(f"Validation loss per fold: {loss_total}")  
    return np.mean(loss_total)



if __name__ == "__main__":
    args = create_train_parser()
    run_prefix:str = args.run_prefix
    b_clean_study:bool = args.b_clean_study
    b_save_checkpoint:bool = args.save_checkpoint
    pretrained:bool = args.pretrained
    b_bilinear:bool = args.b_bilinear
    replace_stride_with_dilation:bool = args.replace_stride_with_dilation
    encoder_name:str = args.encoder_name
    architecture:str = args.architecture
    lr_ranges = [args.lr_min, args.lr_max]

    if args.db_name == "":
        db_name:str = f"{run_prefix}_{architecture}_{encoder_name}_dil{int(replace_stride_with_dilation)}_bilin{int(b_bilinear)}_pre{int(pretrained)}"
    else:
        db_name = args.db_name
    if args.study_name == "":
        study_name:str = f"{architecture}_{encoder_name}_dil{int(replace_stride_with_dilation)}_bilin{int(b_bilinear)}_pre{int(pretrained)}"
    else:
        study_name = args.study_name
    root_path: str = args.root_path
    data_path = Path(root_path) / "data" 
    num_folds:int = args.n_folds
    batch_size:int = args.batch_size
    n_trials:int = args.n_trials

    lr_factor_ranges = [0.1, 0.9, 0.1]
    max_epochs:int = 100
    es_patience:int = 10
    lr_scheduler_patience:int = 5
    seed:int = args.seed

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2

    seed_all(seed=seed)

    # Create Paths
    model_path = Path(f'{root_path}/models/')
    model_path.mkdir(parents=True, exist_ok=True)
    result_path = Path(f'{root_path}/results/')
    result_path.mkdir(parents=True, exist_ok=True)

    study = set_study(db_name=db_name, study_name=study_name, root_path=root_path, seed=seed, b_clean_study=b_clean_study)

    study.optimize(lambda trial: objective(trial), n_trials=n_trials)