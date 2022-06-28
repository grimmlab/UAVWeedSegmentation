#!/bin/bash
cd /home/ngenze/UAVWeedSegmentation
# Run all experiments in one go
# TODO: add more experiments

echo "----- STARTING 1/4 -----"
python3 train.py "fcn8s" "resnet101" --batch_size 100 --pretrained --b_bilinear --replace_stride_with_dilation --run_prefix "save" 
echo "----- STARTING 2/4 -----"
python3 train.py "fcn8s" "resnet50" --batch_size 100 --pretrained --b_bilinear --replace_stride_with_dilation --run_prefix "save" 
echo "----- STARTING 3/4 -----"
python3 train.py "fcn8s" "resnet34" --batch_size 100 --pretrained --b_bilinear --replace_stride_with_dilation --run_prefix "save" 
echo "----- STARTING 4/4 -----"
python3 train.py "fcn8s" "resnet18" --batch_size 100 --pretrained --b_bilinear --replace_stride_with_dilation --run_prefix "save" 
