# ----------------------------------------------
# Sensitivity check of quantizing weights
# ----------------------------------------------
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GAT)
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
                --save_prefix ./sensitivity_check/aggregation \
                --quant \
                --num_wei_bits ${ACT_WEI_BITS[k]} \
                --num_act_bits ${ACT_WEI_BITS[k]} \
                --num_agg_bits ${AGG_BITS[v]} \
                --num_att_bits 32
            done
        done
    done
done