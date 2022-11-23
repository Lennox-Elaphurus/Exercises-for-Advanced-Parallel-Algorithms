#!/bin/bash
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --output=vary_radii.txt

for RADIUS in {1..7}; do
  srun -o vary_radii.txt --open-mode=append FDTD3d --radius=$((RADIUS))
done
