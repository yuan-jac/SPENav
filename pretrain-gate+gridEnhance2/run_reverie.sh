NODE_RANK=0
NUM_GPUS=3
export OMP_NUM_THREADS=4 # 或根据你的CPU核数和负载调整
outdir=../datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.sap.og-init.lxmert-aug.speaker-new-2

# train
CUDA_VISIBLE_DEVICES='5,6,7' torchrun --master_port 29514 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_reverie_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/reverie_obj_model_config.json \
    --config config/reverie_obj_pretrain.json \
    --output_dir $outdir \
    --checkpoint /home/files/A/zhanghuaxiang3/GridMM/datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.sap.og-init.lxmert-aug.speaker-new/ckpts/model_step_17500.pt \