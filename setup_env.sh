#!/bin/bash
#====================================================================
# One-time environment setup for Alpaca-LoRA chatbot lab on Iridis X
# RUN THIS ON THE LOGIN NODE (NOT inside an sbatch job)
#
# Why on the login node?
#   - Compute nodes on Iridis have NO internet access, so pip/conda
#     installs and HuggingFace model downloads MUST happen here.
#   - The login node has no GPU, so the model itself can't be tested
#     here; that part runs from the SLURM script.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#====================================================================

set -e   # exit on first error

#--------------------------------------------------------------------
# 1. Load conda + CUDA modules (same versions as the SLURM job)
#--------------------------------------------------------------------
module purge
module load conda/py3-latest
module load cuda/12.1

#--------------------------------------------------------------------
# 2. Create the conda environment (once)
#    If 'nlp_labs' already exists, this step is skipped.
#--------------------------------------------------------------------
ENV_NAME="nlp_labs"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists - skipping creation."
else
    echo "Creating conda env '${ENV_NAME}' with Python 3.9..."
    conda create --yes -n "${ENV_NAME}" python=3.9
fi

# Activate the env for the rest of this script
source activate "${ENV_NAME}"

#--------------------------------------------------------------------
# 3. Install PyTorch with CUDA 12.1 support
#    (matches CUDA module loaded above)
#--------------------------------------------------------------------
echo "Installing PyTorch (CUDA 12.1)..."
conda install --yes pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    pytorch-cuda=12.1 -c pytorch -c nvidia

#--------------------------------------------------------------------
# 4. Set up the lab directory and unzip lab files
#    Adjust LAB_ROOT if you want it under /scratch instead of $HOME.
#--------------------------------------------------------------------
LAB_ROOT="${HOME}/alpaca_lora_lab"
mkdir -p "${LAB_ROOT}"
cd "${LAB_ROOT}"

# Expectation: you have already SCP'd the two zip files here:
#   alpaca-lora-07-12-2023.zip
#   alpaca_lora_lab_chatbot_package.zip
# from your local machine using:
#   scp file.zip <username>@iridis5_a.soton.ac.uk:~/alpaca_lora_lab/
if [ ! -f "alpaca-lora-07-12-2023.zip" ] || [ ! -f "alpaca_lora_lab_chatbot_package.zip" ]; then
    echo "ERROR: Missing zip files in ${LAB_ROOT}."
    echo "       SCP these files from your local machine first:"
    echo "         alpaca-lora-07-12-2023.zip"
    echo "         alpaca_lora_lab_chatbot_package.zip"
    exit 1
fi

echo "Unzipping lab files..."
unzip -o alpaca-lora-07-12-2023.zip
unzip -o alpaca_lora_lab_chatbot_package.zip

# Drop the lab's generate_chatbot.py into the alpaca-lora repo
cp alpaca_lora_lab_chatbot_package/generate_chatbot.py alpaca-lora/

#--------------------------------------------------------------------
# 5. Install Python dependencies (pinned versions from the lab)
#--------------------------------------------------------------------
echo "Installing pip requirements..."
python3 -m pip install -r alpaca_lora_lab_chatbot_package/alpaca-lora-requirements.txt
python3 -m pip install nltk

# Pre-download NLTK data so the compute node doesn't need internet
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Lab note: revert peft to a specific commit so the model serializes correctly
pip uninstall -y peft
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08

#--------------------------------------------------------------------
# 6. Pre-download the LLaMA 3B model from HuggingFace
#    (compute nodes have no internet - this MUST happen here!)
#--------------------------------------------------------------------
echo "Pre-downloading openlm-research/open_llama_3b_v2 to HF cache..."
python3 - <<'PY'
from transformers import LlamaForCausalLM, LlamaTokenizer
name = "openlm-research/open_llama_3b_v2"
print(f"Caching tokenizer for {name}...")
LlamaTokenizer.from_pretrained(name)
print(f"Caching model weights for {name} (this can take a few minutes)...")
LlamaForCausalLM.from_pretrained(name)
print("Done. Model is now in ~/.cache/huggingface/")
PY

#--------------------------------------------------------------------
# 7. Create the shape-exploration script (Lab Step 2)
#    Copy generate_chatbot.py and inject the shape-inspection code
#    where the lab handout says 'ADD LAB SHAPE CODE HERE'.
#--------------------------------------------------------------------
cd "${LAB_ROOT}/alpaca-lora"

if [ ! -f "generate_chatbot_shapes.py" ]; then
    echo "Creating generate_chatbot_shapes.py with layer/tensor inspection..."
    cp generate_chatbot.py generate_chatbot_shapes.py

    # Insert the shape-exploration block (from the lab handout) using Python
    # so we don't have to rely on fragile sed escaping.
    python3 - <<'PY'
shape_code = '''\t# ===== LAB: explore model architecture and tensor shapes =====
\tprint('\\n\\nmodel architecture:')
\tprint(model)

\t# V = 32000, dim = 3200
\tprint('\\nembedding layer:')
\tprint(model.model.embed_tokens)

\t# 26 decoder layers
\tprint('\\nfirst decoder layer:')
\tprint(model.model.layers[0])

\t# self attention layer = 4 x linear (q,k,v,o) + ROPE
\tprint('\\nfirst self-attention layer:')
\tprint(model.model.layers[0].self_attn)

\t# embedding layer in/out shape
\tprint('\\n\\ntensor input shape (embedding layer):')
\ttest_tensor = torch.LongTensor([1]).to("cuda")
\tprint(test_tensor.size())

\tprint('\\ntensor output shape (embedding layer):')
\ttest_output = model.model.embed_tokens(test_tensor)
\tprint('size = ', test_output.size())
\tprint('embedding (static) tensor for vocab index 1 = ', repr(test_output))

\ttest_tensor2 = torch.LongTensor([2]).to("cuda")
\ttest_output2 = model.model.embed_tokens(test_tensor2)
\tprint('embedding (static) tensor for vocab index 2 = ', repr(test_output2))

\t# self-attention in/out shape
\tprint('\\ntensor input shape (self-attention layer):')
\ttest_tensor = test_output2
\tprint(test_tensor.size())

\tprint('\\ntensor output shape (self-attention layer):')
\ttest_output = model.model.layers[0].self_attn.o_proj(test_tensor)
\tprint(test_output.size())

\t# LM head in/out shape
\tprint('\\ntensor input shape (LM linear layer):')
\ttest_tensor = test_output
\tprint(test_tensor.size())

\tprint('\\ntensor output shape (LM linear layer):')
\ttest_output = model.lm_head(test_tensor)
\tprint(test_output.size())
\tprint('\\n')
\t# ===== END LAB shape code =====
'''

with open("generate_chatbot_shapes.py", "r") as f:
    src = f.read()

marker = "# ADD LAB SHAPE CODE HERE"
if marker in src:
    src = src.replace(marker, marker + "\n" + shape_code)
    with open("generate_chatbot_shapes.py", "w") as f:
        f.write(src)
    print("Inserted shape-inspection code at marker.")
else:
    print("WARNING: marker not found - please add shape code manually.")
PY
fi

#--------------------------------------------------------------------
# 8. Done
#--------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Setup complete. To submit the lab job:"
echo "    cd ${LAB_ROOT}"
echo "    sbatch run_chatbot_lab.slurm"
echo "Then monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f logs/slurm-<jobid>.out"
echo "================================================================"
