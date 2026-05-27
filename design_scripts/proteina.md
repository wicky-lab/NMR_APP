# Reproducing Inference Runs

This document describes the procedure used to generate protein backbones with
the unconditional 400 M-parameter Proteina checkpoint (`proteina_v1.4_D21M_400M_tri.ckpt`)
on a SLURM-managed cluster. It supplements the main `README.md` with the exact
launch configuration, environment, and seeding scheme used to produce the
samples reported in the manuscript.

## 1. Environment

Follow the setup instructions in `README.md` to create the `proteina_env`
conda environment and to populate `.env` with the location of the auxiliary
data files (checkpoints, CATH metadata, FID reference embeddings).

For the runs described here we used:

| Variable        | Value                                                            |
|-----------------|------------------------------------------------------------------|
| `code_dir`      | `/Users/dabramson/Desktop/proteina`                              |
| `DATA_PATH`     | `/cluster/work/wicky/datasets/proteina_additional_files`         |
| GPU             | NVIDIA RTX 4090 (single device)                                  |
| CPUs / job      | 6                                                                |
| Memory / CPU    | 5 GB                                                             |
| Wall-clock      | 12 h                                                             |

## 2. Seeding from `SLURM_ARRAY_TASK_ID`

To guarantee that array tasks explore disjoint regions of the sampler's
noise distribution — and that the full sweep is bit-for-bit reproducible —
the random seed for each task is derived deterministically from the SLURM
array index rather than from the static `seed` field of the Hydra config.

The relevant lines in `proteinfoundation/inference.py` are:

```python
# Set seed to slurm array id or config seed
batch_id = int(os.environ.get("SLURM_ARRAY_TASK_ID")) + 2000
print(f"Batch ID: {batch_id}")
L.seed_everything(batch_id)
```

`L.seed_everything` is the PyTorch Lightning helper that simultaneously
seeds Python's `random` module, NumPy, PyTorch (CPU and CUDA), and sets
`PYTHONHASHSEED`. Because the seed is set *before* the model is moved to
the device and *before* the dataloader iterator is constructed, every
source of stochasticity in the forward pass.


