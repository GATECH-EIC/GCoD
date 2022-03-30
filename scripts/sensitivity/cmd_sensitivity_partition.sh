
# ----------------------------------------------
# Sensitivity check of quantizing node features
# ----------------------------------------------
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GraphSAGE)
ACT_BITS=(32 16 8 6 4 3 2 1)
WEI_BITS=(32 16 8)

for ((i=0; i<1; i++)) # model
do
    for ((j=0; j<3; j++)) # dataset
    do
        for ((k=0; k<3; k++)) # weight bits
        do
            for ((v=0; v<8; v++)) # activation bits
            do
                echo ${MODELS[i]}
                echo ${DATASETS[j]}
                echo ${WEI_BITS[k]}
                echo ${ACT_BITS[v]}

                python train_sage.py \
                --model ${MODELS[i]} \
                --dataset ${DATASETS[j]} \
                --device cuda:1 \
                --save_prefix ./sensitivity_check/activation_sage_partition \
                --quant \
                --partition \
                --num_wei_bits ${WEI_BITS[k]} \
                --num_act_bits ${ACT_BITS[v]}
            done
        done
    done
done

# ----------------------------------------------
# Sensitivity check of quantizing weights
# ----------------------------------------------
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GraphSAGE)
WEI_BITS=(32 16 8 6 4 3 2 1)
ACT_BITS=(32 16 8)

for ((i=0; i<1; i++)) # model
do
    for ((j=0; j<3; j++)) # dataset
    do
        for ((k=0; k<3; k++)) # activation bits
        do
            for ((v=0; v<8; v++)) # weight bits
            do
                echo ${MODELS[i]}
                echo ${DATASETS[j]}
                echo ${ACT_BITS[k]}
                echo ${WEI_BITS[v]}

                python train_sage.py \
                --model ${MODELS[i]} \
                --dataset ${DATASETS[j]} \
                --device cuda:1 \
                --save_prefix ./sensitivity_check/weight_sage_partition \
                --quant \
                --partition \
                --num_wei_bits ${WEI_BITS[v]} \
                --num_act_bits ${ACT_BITS[k]}
            done
        done
    done
done