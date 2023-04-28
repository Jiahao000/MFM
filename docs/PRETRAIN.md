# Pre-training

```shell
bash dist_pretrain.sh ${CONFIG_FILE} ${GPUS} --data-path <imagenet-path>/train [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

`[]` indicates optional arguments that can be found in [main_mfm.py](../main_mfm.py). You can easily modify config options with these arguments.

For example, to pre-train `MFM` using `ViT-Base` as the backbone, run the following on 16 GPUs:

```shell
bash dist_pretrain.sh configs/vit_base/mfm_pretrain__vit_base__img224__300ep.yaml 16 --data-path <imagenet-path>/train --batch-size 128 [--output <output-directory> --tag <job-tag>]
```

Alternatively, if you run `MFM` on a cluster managed with [slurm](https://slurm.schedmd.com/):

```shell
GPUS_PER_NODE=${GPUS_PER_NODE} SRUN_ARGS=${SRUN_ARGS} bash slurm_pretrain.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${GPUS} --data-path <imagenet-path>/train [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

For example, to pre-train `MFM` using `ViT-Base` as the backbone, run the following on 2 nodes with 8 GPUs each:

```shell
# The default setting: GPUS_PER_NODE=8
bash slurm_pretrain.sh Dummy Pretrain_job configs/vit_base/mfm_pretrain__vit_base__img224__300ep.yaml 16 --data-path <imagenet-path>/train --batch-size 128 [--output <output-directory> --tag <job-tag>]
```
