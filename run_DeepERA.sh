#!/bin/bash

cuda=0
radius=2
ngram=3
dim=10
layer_gnn=3
layer_gcn=1
side=5
window=$((2*side+1))
layer_cnn=3
layer_output=3
batch_size=128
lr=1e-3
lr_decay=0.5
decay_interval=20
weight_decay=1e-6
epoch=31
ite=0
for var in "$@"
do
    if [ $ite -lt 2 ]
    then
        ite=$((ite+1))
        continue
    fi
    eval $var
done

CUDA_VISIBLE_DEVICES=$cuda python run_DeepERA.py $batch_size $radius $ngram $dim $layer_gnn $layer_gcn $window $layer_cnn $layer_output $lr $lr_decay $decay_interval $weight_decay $epoch
