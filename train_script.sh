#!/bin/bash
cd /home/ngenze/UAVWeedSegmentation
architecture="fcn16s" 
encoder="resnet50"


# Run all experiments in one go
# TODO: add more experiments

# no dilation, 32x upsampling using resnet50
python3 train.py "fcntv" "resnet50" --pretrained --b_bilinear --run_prefix "save"

# dilation x4, --> 8x upsampling using resnet50
python3 train.py "fcntv" "resnet50" --replace_stride_with_dilation --b_bilinear --pretrained --run_prefix "save"