DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GCN GAT GIN)

for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python plot_adj.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --quant
    done
done