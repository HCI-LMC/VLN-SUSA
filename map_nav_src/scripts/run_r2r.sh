DATA_ROOT=/data/public_datasets/VLN/zxs/datasets
#DATA_ROOT=/dev/shm/zxs/datasets

train_alg=dagger

features=vitbase
ft_dim=768
dft_dim=1000
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

features=clip # vit or clip
depth_feat = glb # no or glb or imageet
sem_feat = blip2 # no or blip2
clloss = 0.1 # no or 0.1,0.2, 0.4, 0.6, 0.8, 1.0
pre = complete # complete or random or partial

name=V${features}-D${depth_feat}-sem${sem_feat}-Loss${clloss}-bypre${pre}
running_time=$(date +%Y%m%d_%Hh%Mm%Ss)
name=${name}-seed.${running_time} # 时间，防止覆盖，但是无用的要及时清理

outdir=../Out/R2R/finetune/t-avg #${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 8
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --depth_feat_size ${dft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# train
#PYTHONPATH=../:$PYTHONPATH CUDA_VISIBLE_DEVICES='3' /home/zhangxuesong/.conda/envs/vlnduet/bin/python r2r/main_nav.py $flag  \
#      --tokenizer bert \
#      --bert_ckpt_file '/data/zxs/Matterport3DSimulator/VLNDUET/part-Out/R2R/pretrain/DUET-clip-original/ckpts/model_step_82500.pt' \
#      --eval_first

# 部分预训练权重：
# /data/zxs/Matterport3DSimulator/VLNDUET/part-Out/R2R/pretrain/DUET-clip-original/ckpts/model_step_82500.pt
# 完全预训练权重

# test
PYTHONPATH=../:$PYTHONPATH CUDA_VISIBLE_DEVICES='3' /home/zhangxuesong/.conda/envs/vlnduet/bin/python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --resume_file /data/zxs/Matterport3DSimulator/VLNDUET/MVP-456-5/Out/R2R/finetune/t-avg/ckpts/best_val_unseen \
      --test --submit