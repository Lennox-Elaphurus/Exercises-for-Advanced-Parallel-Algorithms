#!/bin/bash
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --output=reduction.txt

./threadFenceReduction
