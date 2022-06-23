#!/bin/bash
cd /home/ngenze/UAVWeedSegmentation
architecture="fcn16s" 
encoder="resnet50"


# Run all experiments in one go
# TODO: add more experiments
python3 train.py $architecture $encoder --pretrained