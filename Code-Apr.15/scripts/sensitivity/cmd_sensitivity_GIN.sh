# ----------------------------------------------
# Sensitivity check of quantizing node features
# ----------------------------------------------
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GIN)
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

                python train.py \
                --model ${MODELS[i]} \
                --dataset ${DATASETS[j]} \
                --partition \
                --device cuda:0 \
                --save_prefix ./sensitivity_check/activation_GIN \
                --quant \
                --num_wei_bits ${WEI_BITS[k]} \
                --num_act_bits ${ACT_BITS[v]} \
                --num_att_bits 32
            done
        done
    done
done

# ----------------------------------------------
# Sensitivity check of quantizing weights
# ----------------------------------------------
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GIN)
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

                python train.py \
                --model ${MODELS[i]} \
                --dataset ${DATASETS[j]} \
                --partition \
                --device cuda:0 \
                --save_prefix ./sensitivity_check/weight_GIN \
                --quant \
                --num_wei_bits ${WEI_BITS[v]} \
                --num_act_bits ${ACT_BITS[k]} \
                --num_att_bits 32
            done
        done
    done
done

# ----------------------------------------------
# Sensitivity check of quantizing aggregation
# ----------------------------------------------
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GIN)
AGG_BITS=(32 16 8 6 4 3 2 1)
ACT_WEI_BITS=(32 16 8)

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
                echo ${ACT_WEI_BITS[k]}
                echo ${AGG_BITS[v]}

                python train.py \
                --model ${MODELS[i]} \
                --dataset ${DATASETS[j]} \
                --partition \
                --device cuda:0 \
                --save_prefix ./sensitivity_check/aggregation_GIN \
                --quant \
                --num_wei_bits ${ACT_WEI_BITS[k]} \
                --num_act_bits ${ACT_WEI_BITS[k]} \
                --num_agg_bits ${AGG_BITS[v]} \
                --num_att_bits 32
            done
        done
    done
done