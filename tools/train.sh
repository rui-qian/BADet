#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CUDA_VISIBLE_DEVICES=1 $PYTHON train.py ../configs/car_cfg.py
