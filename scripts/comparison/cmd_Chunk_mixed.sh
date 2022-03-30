DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GCN GAT)
WEI_BITS=(2 3)
Q_MIN=(1 2 3 4 1 2 3)
Q_MAX=(4 4 4 4 3 3 3)

for ((i=0; i<2; i++)) # model
do
    for ((j=0; j<3; j++)) # dataset
    do
        for ((k=0; k<2; k++)) # weight bits
        do
            for ((v=0; v<7; v++)) # q_min / q_max bits
            do

                echo "model: ${MODELS[i]}"
                echo "dataset: ${DATASETS[j]}"
                echo "weight bits: ${WEI_BITS[k]}"
                echo "q_max: ${Q_MAX[v]}"
                echo "q_min: ${Q_MIN[v]}"

                python train.py \
                --model ${MODELS[i]} \
                --dataset ${DATASETS[j]} \
                --partition \
                --device cuda:0 \
                --save_prefix ./comparison/Chunk_Mixed_tuning_Jan.26 \
                --quant \
                --enable_chunk_q \
                --enable_chunk_q_mix \
                --num_wei_bits ${WEI_BITS[k]} \
                --num_act_bits 32 \
                --num_agg_bits 32 \
                --num_att_bits 32 \
                --q_max ${Q_MAX[v]} \
                --q_min ${Q_MIN[v]}
            done
        done
    done
done