#!/usr/env/bin bash
#### Do not use RTX GPU! Not compatible with cuda8.0; use the other gpu instead
CUDA_VISIBLE_DEVICES=3 python train_shape.py
