#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
# we run on the gpu partition and we allocate 2 titanx gpus
SBATCH -p gpu --gres=gpu:titanx:1
#We expect that our program should not run langer than 30 min
#Note that a program will be killed once it exceeds this time!
SBATCH --time=00:30:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
cd samples
python s007_3d_reconstruction.py
python s021_pygpu.py
python futhark_fp.py
python futhark_bp.py
python futhark_SIRT.py
python futhark_SIRT3D.py
