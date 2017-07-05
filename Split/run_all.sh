#!/bin/bash

rm -rf ./log/*


CUDA_VISIBLE_DEVICES='' python3 main.py 0
CUDA_VISIBLE_DEVICES='' python3 main.py 1
CUDA_VISIBLE_DEVICES='' python3 main.py 2

