# Running the Alpaca-LoRA Chatbot Lab on Iridis X

This bundle contains everything you need to run the COMP3225 / COMP6253
"LLM as a chatbot" lab on the **Iridis X** GPU cluster at the University
of Southampton, instead of the 8 GB ECS lab machines.

## Files

| File | Purpose | Where it runs |
|------|---------|---------------|
| `setup_env.sh` | One-time install: conda env, PyTorch, lab deps, downloads the LLaMA 3B model. | **Login node** (needs internet) |
| `run_chatbot_lab.slurm` | The SLURM batch script that actually runs the chatbot on a GPU. | **Compute node** (submitted via `sbatch`) |
| `README.md` | This file. | — |

## Why two scripts and not one?

Iridis compute nodes have **no internet access**. Anything that requires
downloading (pip packages, HuggingFace model weights, NLTK data) must be
done **on the login node first**. The compute node then just *uses* the
cached files. This is the single most common reason ML jobs fail on Iridis,
so the workflow below splits cleanly into "install once" + "submit job".

## Step-by-step

### 1. Connect to Iridis

From outside the campus network you need the University VPN first, then:

```bash
ssh <your_username>@iridis5_a.soton.ac.uk
```

(Iridis X is the GPU service of Iridis 6 but is currently accessed through
the Iridis 5 login nodes for ECS users.)

### 2. Copy the lab zip files to Iridis

From your **local** machine:

```bash
mkdir -p ~/alpaca_lora_lab    # on Iridis (do this after first SSH)

# from your laptop:
scp alpaca-lora-07-12-2023.zip            <user>@iridis5_a.soton.ac.uk:~/alpaca_lora_lab/
scp alpaca_lora_lab_chatbot_package.zip   <user>@iridis5_a.soton.ac.uk:~/alpaca_lora_lab/
scp setup_env.sh run_chatbot_lab.slurm    <user>@iridis5_a.soton.ac.uk:~/alpaca_lora_lab/
```

### 3. Run the one-time setup (login node)

```bash
cd ~/alpaca_lora_lab
chmod +x setup_env.sh
./setup_env.sh
```

This will:
- create the `nlp_labs` conda env with Python 3.9
- install PyTorch 2.2.0 + CUDA 12.1
- install all `alpaca-lora-requirements.txt` deps + the pinned `peft` commit
- download NLTK data
- pre-download the `openlm-research/open_llama_3b_v2` weights to `~/.cache/huggingface/`
- generate `generate_chatbot_shapes.py` with the lab's shape-inspection block

### 4. Edit your email in the SLURM script

Open `run_chatbot_lab.slurm` and change this line to your actual email:

```bash
#SBATCH --mail-user=YOUR_USERNAME@soton.ac.uk
```

### 5. Submit the job

```bash
cd ~/alpaca_lora_lab
sbatch run_chatbot_lab.slurm
```

You'll get back a job ID, e.g. `Submitted batch job 3107069`.

### 6. Monitor

```bash
squeue -u $USER                          # see if it's running / queued
tail -f logs/slurm-3107069.out           # follow live output
scontrol show job 3107069                # detailed job info
scancel 3107069                          # if you need to kill it
```

The output of the **shape-inspection script** (the LAB SIGN-OFF deliverable)
will appear in `logs/slurm-<jobid>.out`. Take a screenshot of that section
for your sign-off.

## SLURM resource choices — quick rationale

| Directive | Value | Why |
|-----------|-------|-----|
| `--partition` | `ecsstudents` | Default GPU partition for taught ECS students. Use `gpu` or `ecsstaff` if you have research access. |
| `--gres=gpu:1` | 1 GPU | The lab uses one model on one GPU. A100 80 GB is plenty. |
| `--cpus-per-task` | 8 | Reasonable for tokenizer + dataloader work. |
| `--mem` | 32 G | Host-side RAM. Model weights live on the GPU; host just needs headroom. |
| `--time` | 2 h | The lab runs in minutes, but partitions punish over-runs less than under-estimates. |

## Things that commonly trip people up

1. **Forgetting to run `setup_env.sh` on the login node first** → the SLURM
   job will fail because it can't download the model.
2. **Wrong partition** → `ecsstudents` is the safe default for taught modules;
   research students may need `ecsstaff` or `gpu`. Check with `sinfo`.
3. **Module versions drift** → if `module load cuda/12.1` says "module not found",
   run `module avail cuda` and pick the closest 12.x; reinstall PyTorch to match.
4. **HuggingFace cache on the wrong filesystem** → if `$HOME` is small, set
   `export HF_HOME=/scratch/$USER/hf_cache` before running setup, and also
   in the SLURM script.

## Quoting the lab's intent

The lab's deliverable, in its own words, is to:
"show screen shot of the `generate_chatbot_shapes.py` output". Everything
else (trying the 7B model, varying the questions in `generate_chatbot.py`)
is exploration on top of that.
