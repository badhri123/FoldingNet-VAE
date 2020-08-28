#!/usr/env/bin bash
#### Do not use RTX GPU! Not compatible with cuda8.0; use the other gpu instead
CUDA_VISIBLE_DEVICES=1 python /home/badri/bns332/foldingnet/src/train.py
