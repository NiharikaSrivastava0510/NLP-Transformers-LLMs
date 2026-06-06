#!/bin/bash
#====================================================================
# SLURM batch script: Alpaca-LoRA LLM Chatbot Lab (COMP3225 / COMP6253)
# Target system : Iridis X (University of Southampton)
# Author        : Niharika
# Description   : Runs generate_chatbot.py and generate_chatbot_shapes.py
#                 on an A100 GPU node to explore an LLM chatbot and its
#                 layer/tensor architecture.
#====================================================================

#-------------------- SLURM resource directives ---------------------
#SBATCH --job-name=alpaca_lora_chatbot       # job name shown in squeue
#SBATCH --partition=ecsstudents              # ECS student GPU partition (change if you have research access, e.g. 'gpu' or 'ecsstaff')
#SBATCH --nodes=1                            # one node only
#SBATCH --ntasks=1                           # one process
#SBATCH --cpus-per-task=8                    # 8 CPU cores for data loading
#SBATCH --gres=gpu:1                         # request 1 GPU (A100 80GB on Iridis X)
#SBATCH --mem=32G                            # 32 GB host RAM (A100 has 80GB device mem; host RAM ~ device + headroom)
#SBATCH --time=02:00:00                      # walltime: 2 hours (lab is short, but allow buffer)
#SBATCH --output=logs/slurm-%j.out           # stdout -> logs/slurm-<jobid>.out
#SBATCH --error=logs/slurm-%j.err            # stderr -> logs/slurm-<jobid>.err
#SBATCH --mail-type=BEGIN,END,FAIL           # email on these events
#SBATCH --mail-user=YOUR_USERNAME@soton.ac.uk  # <-- CHANGE to your email

#--------------------------------------------------------------------
# 0. Setup: print environment info for debugging
#--------------------------------------------------------------------
echo "================================================================"
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Job name       : ${SLURM_JOB_NAME}"
echo "Node           : $(hostname)"
echo "Partition      : ${SLURM_JOB_PARTITION}"
echo "Started at     : $(date)"
echo "Working dir    : $(pwd)"
echo "================================================================"

# Make sure the logs directory exists (in case it wasn't pre-created)
mkdir -p logs

#--------------------------------------------------------------------
# 1. Load required modules
#    Iridis uses 'module' to expose system software. We need conda
#    (for the python env) and CUDA (for the GPU).
#--------------------------------------------------------------------
module purge                                 # start from a clean module set
module load conda/py3-latest                 # gives us conda + python
module load cuda/12.1                        # CUDA toolkit (must match PyTorch build)

# Sanity check: confirm GPU is visible to the node
echo "----- nvidia-smi -----"
nvidia-smi
echo "----------------------"

#--------------------------------------------------------------------
# 2. Activate the conda environment (created once on the login node;
#    see setup_env.sh for first-time install steps)
#--------------------------------------------------------------------
source activate nlp_labs

# Verify python + CUDA visibility from inside the env
echo "Python version: $(python3 -V)"
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

#--------------------------------------------------------------------
# 3. Move into the lab working directory
#    Adjust LAB_DIR if you placed the lab somewhere else (e.g. scratch)
#--------------------------------------------------------------------
LAB_DIR="${HOME}/alpaca_lora_lab/alpaca-lora"
cd "${LAB_DIR}" || { echo "ERROR: cannot cd to ${LAB_DIR}"; exit 1; }
echo "Now in: $(pwd)"

#--------------------------------------------------------------------
# 4. (Optional) GPU memory tuning - mirrors the script's own setting,
#    but exporting here makes it visible to any subprocesses too.
#--------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF='garbage_collection_threshold:0.6,max_split_size_mb:256'

#--------------------------------------------------------------------
# 5. Run the chatbot - Step 1 of the lab
#    Default: 3B LLaMA model. The A100 80GB on Iridis X is far more
#    than 8GB, so you can also try the 7B model if you wish.
#--------------------------------------------------------------------
echo ""
echo "================ Running generate_chatbot.py (3B) ==============="
python3 generate_chatbot.py --base_model 'openlm-research/open_llama_3b_v2'

#--------------------------------------------------------------------
# 6. Run the architecture/shape exploration - Step 2 (LAB SIGN-OFF)
#    Make sure generate_chatbot_shapes.py exists with the
#    'ADD LAB SHAPE CODE HERE' block filled in (see lab handout).
#--------------------------------------------------------------------
if [ -f "generate_chatbot_shapes.py" ]; then
    echo ""
    echo "============ Running generate_chatbot_shapes.py (3B) ============"
    python3 generate_chatbot_shapes.py --base_model 'openlm-research/open_llama_3b_v2'
else
    echo "WARNING: generate_chatbot_shapes.py not found - skipping shape exploration."
    echo "         Create it from generate_chatbot.py and add the lab shape code."
fi

#--------------------------------------------------------------------
# 7. (Optional) Try the 7B model - uncomment if you want to compare
#--------------------------------------------------------------------
# echo ""
# echo "================ Running generate_chatbot.py (7B) ==============="
# python3 generate_chatbot.py --base_model 'openlm-research/open_llama_7b_v2'

#--------------------------------------------------------------------
# 8. Done
#--------------------------------------------------------------------
echo "================================================================"
echo "Finished at: $(date)"
echo "================================================================"
