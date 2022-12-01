#!/bin/bash
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --output=pow2Reduction.txt

for SIZE in {2..40}; do
  srun -o pow2Reduction.txt --open-mode=append ./threadFenceReduction --n=$((2**SIZE))
done
