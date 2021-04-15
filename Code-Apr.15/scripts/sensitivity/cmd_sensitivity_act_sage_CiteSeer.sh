# ===========================
# GraphSAGE
# ===========================


# ----------------------------------------------
# Sensitivity check of quantizing node features
# ----------------------------------------------
MODELS=(GraphSAGE)
ACT_BITS=(32 16 8 6 4 3 2 1)
WEI_BITS=(32 16 8)

for ((i=0; i<1; i++)) # model
do
    for ((k=0; k<3; k++)) # weight bits
    do
        for ((v=0; v<8; v++)) # activation bits
        do
            echo ${MODELS[i]}
            echo ${WEI_BITS[k]}
            echo ${ACT_BITS[v]}

            python train_sage.py \
            --model ${MODELS[i]} \
            --dataset CiteSeer \
            --device cuda:0 \
            --save_prefix ./sensitivity_check/activation_sage_CiteSeer \
            --quant \
            --num_wei_bits ${WEI_BITS[k]} \
            --num_act_bits ${ACT_BITS[v]}
        done
    done
done
