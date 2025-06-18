NODE_RANK=0
NUM_GPUS=1
export OMP_NUM_THREADS=4 # 或根据你的CPU核数和负载调整
export TMPDIR=/home/files/A/zhanghuaxiang3/tmp_cache
outdir=../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.sap-init.lxmert-aug.speaker-new-nock-allmodule-dino200000-2

# train (使用 torchrun)
CUDA_VISIBLE_DEVICES='3' torchrun --master_port 29503 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_pretrain.json \
    --output_dir $outdir \
    --checkpoint /home/files/A/zhanghuaxiang3/GridMM/datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.sap-init.lxmert-aug.speaker-new-nock-allmodule-dino200000/ckpts/model_step_94500.pt \