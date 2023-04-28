# Fine-tuning pre-trained models

## Evaluation

To evaluate a provided model on ImageNet validation set, run:

```shell
bash dist_finetune.sh ${CONFIG_FILE} ${PRETRAIN_CKPT} ${GPUS} --eval --resume <finetuned-ckpt> --data-path <imagenet-path>
```

For example, to evaluate the `ViT-Base` model pre-trained by `MFM` on a single GPU, run:

```shell
bash dist_finetune.sh configs/vit_base/finetune__vit_base__img224__100ep.yaml <pretrained-ckpt> 1 --eval --resume <finetuned-ckpt> --data-path <imagenet-path>
```

Alternatively, if you run `MFM` on a cluster managed with [slurm](https://slurm.schedmd.com/):

```shell
GPUS_PER_NODE=${GPUS_PER_NODE} SRUN_ARGS=${SRUN_ARGS} bash slurm_finetune.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${PRETRAIN_CKPT} ${GPUS} --eval --resume <finetuned-ckpt> --data-path <imagenet-path>
```

For example, to evaluate the `ViT-Base` model pre-trained by `MFM` on a single GPU, run:

```shell
GPUS_PER_NODE=1 bash slurm_finetune.sh Dummy Test_job configs/vit_base/finetune__vit_base__img224__100ep.yaml <pretrained-ckpt> 1 --eval --resume <finetuned-ckpt> --data-path <imagenet-path>
```

## Fine-tuning

To fine-tune models pre-trained by `MFM`, run:

```shell
bash dist_finetune.sh ${CONFIG_FILE} ${PRETRAIN_CKPT} ${GPUS} --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

`[]` indicates optional arguments that can be found in [main_finetune.py](../main_finetune.py). You can easily modify config options with these arguments.

For example, to fine-tune `ViT-Base` pre-trained by `MFM`, run the following on 16 GPUs:

```shell
bash dist_finetune.sh configs/vit_base/finetune__vit_base__img224__100ep.yaml  <pretrained-ckpt> 16 --data-path <imagenet-path> --batch-size 128 [--output <output-directory> --tag <job-tag>]
```

Alternatively, if you run `MFM` on a cluster managed with [slurm](https://slurm.schedmd.com/):

```shell
GPUS_PER_NODE=${GPUS_PER_NODE} SRUN_ARGS=${SRUN_ARGS} bash slurm_finetune.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${PRETRAIN_CKPT} ${GPUS} --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

For example, to fine-tune `ViT-Base` pre-trained by `MFM`, run the following on 2 nodes with 8 GPUs each:

```shell
# The default setting: GPUS_PER_NODE=8
bash slurm_finetune.sh Dummy Finetune_job configs/vit_base/finetune__vit_base__img224__100ep.yaml <pretrained-ckpt> 16 --data-path <imagenet-path> --batch-size 128 [--output <output-directory> --tag <job-tag>]
```
