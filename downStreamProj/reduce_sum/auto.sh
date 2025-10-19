#!/bin/bash

# 指定只使用 GPU 0 (NVIDIA GeForce RTX 3090)
# 这样可以确保程序和cuda-gdb都只在3090上运行，不受4060 Ti影响
export CUDA_VISIBLE_DEVICES=0

make clean && make
# outputs/app 2 64 64 64
# outputs/app 2 10240 256 10240

# make clean && make && clear && outputs/app 1 1024 256 1024
