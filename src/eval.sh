#!/usr/env/bin bash
actv folding
CUDA_VISIBLE_DEVICES=1 python eval.py ../model/ep_15.pth
