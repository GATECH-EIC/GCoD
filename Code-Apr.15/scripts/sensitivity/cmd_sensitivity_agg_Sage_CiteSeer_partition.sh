# ===========================
# GraphSAGE
# ===========================


# ----------------------------------------------
# Sensitivity check of quantizing node features
# ----------------------------------------------
MODELS=(GraphSAGE)
AGG_BITS=(32 16 8 6 4 3 2 1)
ACT_WEI_BITS=(32 16 8)

for ((i=0; i<1; i++)) # model
do
    for ((k=0; k<3; k++)) # weight bits
    do
        for ((v=0; v<8; v++)) # activation bits
        do
            echo ${MODELS[i]}
            echo ${ACT_WEI_BITS[k]}
            echo ${AGG_BITS[v]}

            python train_sage.py \
            --model ${MODELS[i]} \
            --dataset CiteSeer \
            --device cuda:0 \
            --save_prefix ./sensitivity_check/aggregation_sage_CiteSeer_partition \
            --quant \
            --partition \
            --num_wei_bits ${ACT_WEI_BITS[k]} \
            --num_act_bits ${ACT_WEI_BITS[k]} \
            --num_agg_bits ${AGG_BITS[v]}
        done
    done
done
