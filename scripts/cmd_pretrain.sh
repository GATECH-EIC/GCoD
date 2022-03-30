DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GCN GAT)

for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python train.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --partition \
        --device cuda:0 \
        --save_prefix pretrain_partition
    done
done

for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python train.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --partition \
        --device cuda:0 \
        --save_prefix pretrain_partition \
        --quant \
        --num_bits 16
    done
done

for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python train.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --partition \
        --device cuda:0 \
        --save_prefix pretrain_partition \
        --quant \
        --num_bits 8
    done
done

for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python train.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --partition \
        --device cuda:0 \
        --save_prefix pretrain_partition \
        --quant \
        --num_bits 4
    done
done

for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python train.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --partition \
        --device cuda:0 \
        --save_prefix pretrain_partition \
        --quant \
        --num_bits 3
    done
done

for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python train.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --partition \
        --device cuda:0 \
        --save_prefix pretrain_partition \
        --quant \
        --num_bits 2
    done
done

for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python train.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --partition \
        --device cuda:0 \
        --save_prefix pretrain_partition \
        --quant \
        --num_bits 1
    done
done

# for ((i=0; i<3; i++))
# do
#     python train_sage.py \
#     --model GraphSAGE \
#     --dataset ${DATASETS[i]} \
#     --partition \
#     --device cuda:0 \
#     --save_prefix pretrain_partition
# done