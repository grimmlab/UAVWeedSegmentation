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
    get_loadersh5,
    train_epoch,
    validate_epoch,
)
from utils.parser import create_train_parser

num_folds = 4
data_path = 

architecture = "fcn8s"
encoder_name = "resnet18"
pretrained = True
b_bn=True
b_bilinear=True
replace_stride_with_dilation=True
device="cuda"

loss_total = np.ones(num_folds)*99999
epochs = np.ones(num_folds)*0
img_list, msk_list = get_patch_lists(
    data_path=data_path, 
    subset="trainval")

fold =0
max_epochs=10

model = set_model(architecture=architecture, encoder_name=encoder_name, pretrained=pretrained, b_bn=b_bn, b_bilinear=b_bilinear, replace_stride_with_dilation=replace_stride_with_dilation, num_classes=3).to(device=device)
        
lr = 1e-4
reduce_factor = 0.1
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=reduce_factor, min_lr=1e-7, patience=5)
means, stds = get_calculated_means_stds_per_fold(fold)
train_loader, valid_loader = get_loaders(
    train_img_dir = train_img_dir,
    train_msk_dir = train_msk_dir,
    valid_img_dir = valid_img_dir, 
    valid_msk_dir = valid_msk_dir,
    mean = means,
    std = stds,
    batch_size = 130,
    num_workers = 2,
    pin_memory = True,
)

scaler = torch.cuda.amp.GradScaler()
for epoch in range(max_epochs):
    train_loss = train_epoch(
        train_loader, 
        model, 
        optimizer, 
        scaler, 
        cur_epoch=epoch,
        trial_number=0,
        fold=fold,
        architecture=architecture
        )

    valid_loss = validate_epoch(
        valid_loader, 
        model, 
        cur_epoch=epoch, 
        trial_number=0,
        fold=fold,
        architecture=architecture
        )
    scheduler.step(valid_loss)

    print(f"{train_loss=}, {valid_loss=}")