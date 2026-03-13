NODE_RANK=0
NUM_GPUS=1
OUTDIR=../datasets/SOON/exprs_map/pretrain/cmt-dino.mlm.sap.og-init.lxmert-new2

CUDA_VISIBLE_DEVICES=1 torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --node_rank=${NODE_RANK} \
    --standalone \
    --master_port=29513 \
    train_soon_obj.py \
    --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/soon_obj_model_config.json \
    --config config/soon_obj_pretrain.json \
    --output_dir ${OUTDIR} \
    --checkpoint /home/files/A/zhanghuaxiang3/GridMM/datasets/SOON/exprs_map/pretrain/cmt-vitbase.butdobj-mlm.sap.og-init.lxmert-new/ckpts/model_step_36000.pt