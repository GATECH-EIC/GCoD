# ===========================
# GraphSAGE
# ===========================


# ----------------------------------------------
# Sensitivity check of quantizing weights
# ----------------------------------------------
MODELS=(GraphSAGE)
WEI_BITS=(32 16 8 6 4 3 2 1)
ACT_BITS=(32 16 8)

for ((i=0; i<1; i++)) # model
do
    for ((k=0; k<3; k++)) # activation bits
    do
        for ((v=0; v<8; v++)) # weight bits
        do
            echo ${MODELS[i]}
            echo ${ACT_BITS[k]}
            echo ${WEI_BITS[v]}

            python train_sage.py \
            --model ${MODELS[i]} \
            --dataset CiteSeer \
            --device cuda:0 \
            --save_prefix ./sensitivity_check/weight_sage_CiteSeer \
            --quant \
            --num_wei_bits ${WEI_BITS[v]} \
            --num_act_bits ${ACT_BITS[k]}
        done
    done
done