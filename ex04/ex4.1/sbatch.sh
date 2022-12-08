#!/bin/bash
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --output=hist.txt

for WC in 1 2 4; do
  for BINS in 256 512 1024 2048 4096 8192; do
    srun -o hist_${WC}.txt --open-mode=append ./histogram --wc=${WC} --binNum=${BINS}
  done
done
