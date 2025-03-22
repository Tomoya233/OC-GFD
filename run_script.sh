#!/bin/bash

dataset_lst=("yelp" "amazon")

lr_lst=(0.001 0.0003 0.0001)
gmm_list=(3 5 7 9)
dropout_lst=(0.5 0.3 0.2 0.0)
early_stop=1

for dataset in "${dataset_lst[@]}"; do
  if [ "$dataset" = "amazon" ]; then 
    hidden_channels_lst=(4 8 16 32)
  else 
    hidden_channels_lst=(4 8 16 32)
  fi
  for gmm_k in "${gmm_list[@]}"; do
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for lr in "${lr_lst[@]}"; do
          for dropout in "${dropout_lst[@]}"; do
            python main.py --dataset $dataset --dropout $dropout\
                  --method multisage --num_layers 2 --hidden_channels $hidden_channels --lr $lr --early_stop $early_stop\
                  --classification GMM_mod --Mahalanobis diagonal_nequal --Measurement mixed --gpu_select 5 --gmm_K $gmm_k
done
done            
done
done 
done