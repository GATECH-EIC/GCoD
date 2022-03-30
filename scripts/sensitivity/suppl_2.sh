
# ----------------------------------------------
# Sensitivity check of attention module
# ----------------------------------------------
# ATT_BITS=(4 3 2 1)

# for ((v=0; v<4; v++))
# do
#     python train.py \
#     --model GAT \
#     --dataset Cora \
#     --partition \
#     --device cuda:6 \
#     --save_prefix ./sensitivity_check/attention \
#     --quant \
#     --num_wei_bits 8 \
#     --num_act_bits 8 \
#     --num_att_bits ${ATT_BITS[v]}
# done

# ATT_BITS=(32 16 8 6 4)

# for ((v=0; v<5; v++))
# do
#     python train.py \
#     --model GAT \
#     --dataset CiteSeer \
#     --partition \
#     --device cuda:6 \
#     --save_prefix ./sensitivity_check/attention \
#     --quant \
#     --num_wei_bits 32 \
#     --num_act_bits 32 \
#     --num_att_bits ${ATT_BITS[v]}
# done

# ----------------------------------------------
# Sensitivity check of activation module
# ----------------------------------------------

# MODELS=(GraphSAGE)
# ACT_BITS=(32 16 8 6 4 3 2 1)
# WEI_BITS=(8)

# for ((i=0; i<1; i++)) # model
# do
#     for ((k=0; k<1; k++)) # weight bits
#     do
#         for ((v=0; v<8; v++)) # activation bits
#         do
#             echo ${MODELS[i]}
#             echo ${WEI_BITS[k]}
#             echo ${ACT_BITS[v]}

#             python train_sage.py \
#             --model ${MODELS[i]} \
#             --dataset Pubmed \
#             --device cuda:0 \
#             --save_prefix ./sensitivity_check/activation_sage_Pubmed \
#             --quant \
#             --num_wei_bits ${WEI_BITS[k]} \
#             --num_act_bits ${ACT_BITS[v]}
#         done
#     done
# done


# ----------------------------------------------
# Sensitivity check of quantizing weights
# ----------------------------------------------
MODELS=(GraphSAGE)
WEI_BITS=(32 16 8 6 4 3 2 1)
ACT_BITS=(8)

for ((i=0; i<1; i++)) # model
do
    for ((k=0; k<1; k++)) # activation bits
    do
        for ((v=0; v<8; v++)) # weight bits
        do
            echo ${MODELS[i]}
            echo ${ACT_BITS[k]}
            echo ${WEI_BITS[v]}

            python train_sage.py \
            --model ${MODELS[i]} \
            --dataset Pubmed \
            --device cuda:0 \
            --save_prefix ./sensitivity_check/weight_sage_Pubmed \
            --quant \
            --num_wei_bits ${WEI_BITS[v]} \
            --num_act_bits ${ACT_BITS[k]}
        done
    done
done