# ===========================
# GCN & GAT
# ===========================
DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GCN GAT)
BITS=(32 16 8 6 4 3 2 1)

for ((i=0; i<2; i++)) # model
do
    for ((j=0; j<3; j++)) # dataset
    do
        for ((k=0; k<8; k++)) # bits
        do
            echo ${MODELS[i]}
            echo ${DATASETS[j]}
            echo ${BITS[k]}

            python train.py \
            --model ${MODELS[i]} \
            --dataset ${DATASETS[j]} \
            --partition \
            --device cuda:0 \
            --save_prefix ./comparison/QAT_Jan.26 \
            --quant \
            --num_wei_bits ${BITS[k]} \
            --num_act_bits ${BITS[k]} \
            --num_agg_bits ${BITS[k]} \
            --num_att_bits 32
        done
    done
done

# ===========================
# GraphSAGE
# ===========================

DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GraphSAGE)
BITS=(32 16 8 6 4 3 2 1)

for ((j=0; j<3; j++)) # dataset
do
    for ((k=0; k<8; k++)) # weight bits
    do
        echo 'GraphSAGE'
        echo ${DATASETS[j]}
        echo ${BITS[k]}

        python train_sage.py \
        --model GraphSAGE \
        --dataset ${DATASETS[j]} \
        --device cuda:2 \
        --save_prefix ./comparison/QAT_Jan.26 \
        --quant \
        --num_wei_bits ${BITS[k]} \
        --num_act_bits ${BITS[k]} \
        --num_agg_bits ${BITS[k]}
    done
done
