# Reproducing Inference Runs
`run_diffusion.sh`:

```bash
/Users/dabramson/Desktop/RFdiffusion/scripts/run_inference.py \
   inference.output_prefix=$output_dir/uncoditional_$SLURM_ARRAY_TASK_ID \
   inference.num_designs=1000 \
   inference.ckpt_override_path=/cluster/work/wicky/params/RFParams/Base_ckpt.pt \
   inference.schedule_directory_path=/Users/dabramson/Desktop/RFdiffusion \
   inference.write_trajectory=False \
   'contigmap.contigs=[100-100]'
```

`$SLURM_ARRAY_TASK_ID` is appended to `inference.output_prefix` so each
array task writes its 1 000 designs to a unique set of PDBs. Seed is automatically random for RFDiffusion
