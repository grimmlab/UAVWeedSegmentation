import logging
import numpy as np
import torch
from tqdm import tqdm
import kornia
from utils.dataset import UAVDatasetPatches
from torch.utils.data import DataLoader
import random
import os
import optuna
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.manual_fcn import load_fcn_resnet
from utils.manual_unet import UNet
from utils.manual_dlplus import DLv3plus

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def get_calculated_means_stds_per_fold(fold):
    means = [
        [0.4895940504368177, 0.4747875829353402, 0.42545172025367883],
        [0.4909516814094245, 0.47507395584447076, 0.4252166750637278],
        [0.4863172918463077, 0.4720067749001233, 0.42307293323046524],
        [0.48556443799258586, 0.471592906257259, 0.42337851381822833]
        ]
    stds = [
        [0.1329905783602554, 0.130645279821384, 0.12234299715980072],
        [0.12910633924968123, 0.12635436744763892, 0.1180632138245313],
        [0.1329739900037901, 0.1304754029316029, 0.12181500603654097],
        [0.1335583288658572, 0.1313047051909438, 0.12297522870807812]
    ]
    return means[fold], stds[fold]

def get_calculated_means_stds_trainval():
    means = [0.48810686542128406, 0.4733653049842984, 0.4242799605915251]
    stds = [0.1321881434144248, 0.12971921686190743, 0.12131885037092494]
    return means, stds

def get_patch_lists(data_path, subset):
    path = data_path / subset / "patches"
    imgPaths = list(path.glob('./img/*.png'))
    img_list = sorted(imgPaths)
    annPaths = list(path.glob('./msk/*.png'))
    msk_list = sorted(annPaths)
    return img_list, msk_list 

def set_study(db_name, study_name, root_path, seed, b_clean_study=False):
    '''
    Creates a new study in a sqlite database located in ./results/
    '''
    sampler = optuna.samplers.TPESampler(seed=seed)
    storage = optuna.storages.RDBStorage(f"sqlite:///{root_path}/results/{db_name}.db", heartbeat_interval=1)
    if b_clean_study:
        print(f"CAUTION: Deleting existing trials in study {study_name}")
        optuna.delete_study(study_name=study_name, storage=f"sqlite:///{root_path}/results/{db_name}.db")
        
    study = optuna.create_study(storage=storage, study_name=study_name, sampler=sampler, direction="minimize", load_if_exists=True)
    return study

def seed_all(seed):
    '''
    sets the initial seed for numpy and pytorch to get reproducible results. 
    One still need to restart the kernel to get reproducible results, as discussed in:
    https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_loaders(train_img_dir, train_msk_dir, valid_img_dir ,valid_msk_dir, mean, std, batch_size, num_workers=4, pin_memory=True):
    train_transform = A.Compose(
        [    
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.CLAHE(),
            A.RandomRotate90(),
            A.Transpose(),
            A.Normalize(
                mean = mean,
                std = std,
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    valid_transform = A.Compose(
        [
            A.Normalize(
                mean = mean,
                std = std,
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    train_ds = UAVDatasetPatches(img_list=train_img_dir, msk_list=train_msk_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    valid_ds = UAVDatasetPatches(img_list=valid_img_dir, msk_list=valid_msk_dir, transform=valid_transform)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, valid_loader
    
def set_model(architecture, encoder_name, pretrained, b_bilinear, replace_stride_with_dilation, num_classes=3):
    model_name = f"{architecture}_{encoder_name}"
    print(f"MODEL NAME: {model_name}")
    if architecture == "fcn32s":
        if replace_stride_with_dilation:
            model=load_fcn_resnet(encoder_name, 
            num_classes=num_classes, 
            pretrained = pretrained, 
            replace_stride_with_dilation=replace_stride_with_dilation, 
            n_upsample=8, 
            b_bilinear=b_bilinear
            )
        else:
            model=load_fcn_resnet(encoder_name, 
            num_classes=num_classes, 
            pretrained = pretrained, 
            replace_stride_with_dilation=replace_stride_with_dilation, 
            n_upsample=32, 
            b_bilinear=b_bilinear
            )

    elif architecture == "fcn16s":
        model=load_fcn_resnet(encoder_name, 
        num_classes=num_classes, 
        pretrained = pretrained, 
        replace_stride_with_dilation=replace_stride_with_dilation, 
        n_upsample=16, 
        b_bilinear=b_bilinear
        )
    elif architecture == "fcn8s":
        model=load_fcn_resnet(encoder_name, 
        num_classes=num_classes, 
        pretrained = pretrained, 
        replace_stride_with_dilation=replace_stride_with_dilation, 
        n_upsample=8, 
        b_bilinear=b_bilinear
        )
    elif architecture == "unet":
        model = UNet(encoder_name=encoder_name)
    
    elif architecture == "dlplus":
        model = DLv3plus(encoder_name=encoder_name, encoder_output_stride=8)
    else:
        raise NotImplementedError("Specified Model is not defined. Currently implemented architectures are: fcn, deeplabv3. Currently implemented feature extractors: resnet50, resnet101")
    return model

def save_checkpoint(state, filename="my_ckpt.pth.tar"):
    torch.save(state, filename)
    return

def train_epoch(loader, model, optimizer, loss_fn, scaler, trial_number=None, fold=None, cur_epoch=None):
    with tqdm(loader, unit="batch", leave=True) as tepoch:
        losses = []
        if fold is not None and trial_number is not None:
            tepoch.set_description(f"Training T{trial_number} F{fold} E{cur_epoch}")
        else:
            tepoch.set_description(f"Retraining E{cur_epoch}")
        for data, targets in tepoch:
            data = data.float().to(device="cuda")
            targets = targets.long().to(device="cuda")
            # forward 
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(True):
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)
                # backward
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # update loop
                tepoch.set_postfix(train_loss=loss.item())
                losses.append(loss.item())
            tepoch.set_postfix(train_losses=np.array(losses).mean())
    return loss.item()

def validate_epoch(loader, model, cur_epoch, fold=None, trial_number=None):
    dice_loss = 0
    predictions_whole = None 
    targets_whole = None 
    model.eval()
    with torch.no_grad():
        with tqdm(loader, unit="batch", leave=False) as tepoch:
            if fold is not None and trial_number is not None:
                tepoch.set_description(f"Validating T{trial_number} F{fold} E{cur_epoch}")
            else:
                tepoch.set_description(f"Validating E{cur_epoch}")
            for idx, (inputs, targets) in enumerate(tepoch):
                inputs = inputs.float().to(device="cuda")
                targets = targets.long().to(device="cuda")
                predictions = model(inputs)
                if predictions_whole is None:
                    predictions_whole = predictions
                else:
                    predictions_whole = torch.cat((predictions_whole, predictions), dim=0)
                if targets_whole is None:
                    targets_whole = targets
                else:
                    targets_whole = torch.cat((targets_whole, targets), dim=0)
                
                
            dice_loss = kornia.losses.dice_loss(predictions_whole, targets_whole).item()
    logging.info(f"Validating T{trial_number} F{fold} E{cur_epoch}: valid loss {dice_loss}")
    model.train()
    return dice_loss

