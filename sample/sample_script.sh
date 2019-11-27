#!/bin/bash

mkdir -p result
python LSTMLM_train.py --model model.h5 --dataset weathernews.npz --batch_size 128 --latent_node 128 --epoch 0
# python LSTMLM_generate.py --model model.h5 --dataset weathernews.npz --format word --size 10
for i in `seq 10`
do
  i_x100=`expr ${i} \* 100`
  python LSTMLM_train.py --model model.h5 --dataset weathernews.npz --batch_size 128 --latent_node 128 --epoch 100 --continue_train
  python LSTMLM_generate.py --model model.h5 --dataset weathernews.npz --format word --size 100 > result/ITER_${i_x100}.txt
done
