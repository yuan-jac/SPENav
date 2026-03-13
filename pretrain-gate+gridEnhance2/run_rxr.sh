#!/bin/bash

NODE_RANK=0
NUM_GPUS=1

outdir=../datasets/RXR/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new

CUDA_VISIBLE_DEVICES='4' torchrun \
  --master_port=29513 \
  --nproc_per_node=${NUM_GPUS} \
  --node_rank=${NODE_RANK} \
  train_rxr.py \
  --world_size ${NUM_GPUS} \
  --vlnbert cmt \
  --model_config config/rxr_model_config.json \
  --config config/rxr_pretrain.json \
  --output_dir ${outdir}
