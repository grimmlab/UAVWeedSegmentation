#!/bin/bash
cd /home/ngenze/UAVWeedSegmentation
# Run all experiments in one go
# TODO: add more experiments

# no dilation, 32x upsampling using resnet50

python3 train.py "unetsmp" "resnet18" --pretrained --run_prefix "check"

# dilation x4, --> 8x upsampling using resnet50
python3 train.py "unetown" "resnet18" --pretrained --run_prefix "check"