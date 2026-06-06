#!/bin/bash
#SBATCH --job-name=bert_imdb
#SBATCH --output=logs/bert_imdb_%j.out
#SBATCH --error=logs/bert_imdb_%j.err
#SBATCH --partition=ecsstudents          # ECS taught-student partition on IridisX
#SBATCH --gres=gpu:1                     # request 1 GPU (L4 24GB by default on this partition)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00                  # 2 hours: plenty for 5k samples x 5 epochs on an L4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_USERNAME@soton.ac.uk    # <-- CHANGE THIS

# --------------------------------------------------------------------------
# Notes on partitions / GPU choice:
#   - ecsstudents : default for ECS students (L4 24GB GPUs). Use this.
#   - swarm       : research staff only, A100/H100 80GB. Don't submit here.
#   - ecsall      : scavenger queue, jobs may be preempted. Useful as backup.
# To request a specific GPU type instead of the default, e.g.:
#   #SBATCH --gres=gpu:a100:1
# --------------------------------------------------------------------------

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Job name:     $SLURM_JOB_NAME"
echo "Node:         $SLURMD_NODENAME"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "Started at:   $(date)"
echo "Working dir:  $(pwd)"
echo "=========================================="

mkdir -p logs

# Load conda and activate the environment created on the login node
module purge
module load conda/py3-latest
source activate bert-imdb-env       # <-- match the env name you created

# Show what we're working with
echo ""
echo "--- GPU info ---"
nvidia-smi
echo ""
echo "--- Python / PyTorch ---"
python -c "import torch; print('torch =', torch.__version__, '| cuda =', torch.cuda.is_available(), '| device =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ""

# Force HuggingFace to use the cache populated on the login node.
# Compute nodes have no internet access, so anything not pre-cached will fail.
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Run the training script
echo "--- Starting training ---"
srun python bert_seq_classifier_imdb.py
echo ""
echo "Finished at: $(date)"
