# ----------------------------------------------
# Sensitivity check of quantizing node features
# ----------------------------------------------
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GCN GAT)
ACT_BITS=(32 16 8 6 4 3 2 1)
WEI_BITS=(32 16 8)

for ((i=0; i<2; i++)) # model
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
                --save_prefix ./sensitivity_check/activation \
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
MODELS=(GCN GAT)
WEI_BITS=(32 16 8 6 4 3 2 1)
ACT_BITS=(32 16 8)

for ((i=0; i<2; i++)) # model
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
                --save_prefix ./sensitivity_check/weight \
                --quant \
                --num_wei_bits ${WEI_BITS[v]} \
                --num_act_bits ${ACT_BITS[k]} \
                --num_att_bits 32
            done
        done
    done
done

# ----------------------------------------------
# Sensitivity check of attention module
# ----------------------------------------------
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GAT)
ATT_BITS=(32 16 8 6 4 3 2 1)
ACT_WEI_BITS=(32 16 8)

for ((i=0; i<1; i++)) # model
do
    for ((j=0; j<3; j++)) # dataset
    do
        for ((k=0; k<3; k++)) # activation / weights bits
        do
            for ((v=0; v<8; v++)) # attention bits
            do
                echo ${MODELS[i]}
                echo ${DATASETS[j]}
                echo ${ACT_WEI_BITS[k]}
                echo ${ATT_BITS[v]}

                python train.py \
                --model ${MODELS[i]} \
                --dataset ${DATASETS[j]} \
                --partition \
                --device cuda:0 \
                --save_prefix ./sensitivity_check/attention \
                --quant \
                --num_wei_bits ${ACT_WEI_BITS[k]} \
                --num_act_bits ${ACT_WEI_BITS[k]} \
                --num_att_bits ${ATT_BITS[v]}
            done
        done
    done
done


# ===========================
# GraphSAGE
# ===========================


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
                --device cuda:0 \
                --save_prefix ./sensitivity_check/activation_sage \
                --quant \
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
                --device cuda:0 \
                --save_prefix ./sensitivity_check/weight_sage \
                --quant \
                --num_wei_bits ${WEI_BITS[v]} \
                --num_act_bits ${ACT_BITS[k]}
            done
        done
    done
done