cd /home/files/A/zhanghuaxiang3/GridMM
export PYTHONPATH=$(pwd)
export PATH=/home/files/A/zhanghuaxiang3/anaconda3/envs/vlnmamba/bin:$PATH
DATA_ROOT=datasets

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}-new-3a

outdir=${DATA_ROOT}/REVERIE/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200
      --max_objects 20

      --batch_size 4
      --lr 2e-5
      --iters 50000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# train
#CUDA_VISIBLE_DEVICES='1,2,3' python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=${ngpus}  main_nav_obj.py $flag  \
#      --tokenizer bert \
#      --bert_ckpt_file ../datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap.og-init.lxmert-aug.speaker-new/ckpts/model_step_25000.pt
      #--eval_first1

# test
CUDA_VISIBLE_DEVICES='4' python3  -m torch.distributed.launch --master_port 29523 --nproc_per_node=${ngpus}  map_nav_src_fullmodule/main_nav_obj.py $flag  \
      --tokenizer bert\
      --bert_ckpt_file  datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.sap.og-init.lxmert-aug.speaker-new-2/ckpts/model_step_52500.pt \
      #--eval_first
      #--resume_file  datasets/REVERIE/exprs_map/finetune/dagger-vitbase-seed.0-new-1/ckpts/best_val_unseen \

      #--test --submit