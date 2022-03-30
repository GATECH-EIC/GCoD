DATASETS=(Cora CiteSeer Pubmed)
MODELS=(GCN GAT)

#------------------------
# No quantizaiton
#------------------------

# Cora & CiteSeer
for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python tune.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --hard \
        --device cuda:4 \
        --save_prefix graph_tune \
        --iteration 1 \
        --ratio_graph 10
    done
done

# Pubmed
for ((i=0; i<2; i++))
do
    echo ${MODELS[i]}
    echo ${DATASETS[2]}
    python tune.py \
    --model ${MODELS[i]} \
    --dataset ${DATASETS[2]} \
    --hard \
    --device cpu \
    --save_prefix graph_tune \
    --iteration 1 \
    --ratio_graph 10
done

#------------------------
# 16 bits
#------------------------

# Cora & CiteSeer
for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python tune.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --hard \
        --device cuda:4 \
        --save_prefix graph_tune \
        --iteration 1 \
        --ratio_graph 10 \
        --quant \
        --num_bits 16
    done
done

# Pubmed
for ((i=0; i<2; i++))
do
    echo ${MODELS[i]}
    echo ${DATASETS[2]}
    python tune.py \
    --model ${MODELS[i]} \
    --dataset ${DATASETS[2]} \
    --hard \
    --device cpu \
    --save_prefix graph_tune \
    --iteration 1 \
    --ratio_graph 10 \
    --quant \
    --num_bits 16
done

#------------------------
# 8 bits
#------------------------

# Cora & CiteSeer
for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python tune.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --hard \
        --device cuda:4 \
        --save_prefix graph_tune \
        --iteration 1 \
        --ratio_graph 10 \
        --quant \
        --num_bits 8
    done
done

# Pubmed
for ((i=0; i<2; i++))
do
    echo ${MODELS[i]}
    echo ${DATASETS[2]}
    python tune.py \
    --model ${MODELS[i]} \
    --dataset ${DATASETS[2]} \
    --hard \
    --device cpu \
    --save_prefix graph_tune \
    --iteration 1 \
    --ratio_graph 10 \
    --quant \
    --num_bits 8
done

#------------------------
# 4 bits
#------------------------

# Cora & CiteSeer
for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python tune.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --hard \
        --device cuda:4 \
        --save_prefix graph_tune \
        --iteration 1 \
        --ratio_graph 10 \
        --quant \
        --num_bits 4
    done
done

# Pubmed
for ((i=0; i<2; i++))
do
    echo ${MODELS[i]}
    echo ${DATASETS[2]}
    python tune.py \
    --model ${MODELS[i]} \
    --dataset ${DATASETS[2]} \
    --hard \
    --device cpu \
    --save_prefix graph_tune \
    --iteration 1 \
    --ratio_graph 10 \
    --quant \
    --num_bits 4
done

#------------------------
# 3 bits
#------------------------

# Cora & CiteSeer
for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python tune.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --hard \
        --device cuda:4 \
        --save_prefix graph_tune \
        --iteration 1 \
        --ratio_graph 10 \
        --quant \
        --num_bits 3
    done
done

# Pubmed
for ((i=0; i<2; i++))
do
    echo ${MODELS[i]}
    echo ${DATASETS[2]}
    python tune.py \
    --model ${MODELS[i]} \
    --dataset ${DATASETS[2]} \
    --hard \
    --device cpu \
    --save_prefix graph_tune \
    --iteration 1 \
    --ratio_graph 10 \
    --quant \
    --num_bits 3
done

#------------------------
# 2 bits
#------------------------

# Cora & CiteSeer
for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python tune.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --hard \
        --device cuda:4 \
        --save_prefix graph_tune \
        --iteration 1 \
        --ratio_graph 10 \
        --quant \
        --num_bits 2
    done
done

# Pubmed
for ((i=0; i<2; i++))
do
    echo ${MODELS[i]}
    echo ${DATASETS[2]}
    python tune.py \
    --model ${MODELS[i]} \
    --dataset ${DATASETS[2]} \
    --hard \
    --device cpu \
    --save_prefix graph_tune \
    --iteration 1 \
    --ratio_graph 10 \
    --quant \
    --num_bits 2
done

#------------------------
# 1 bits
#------------------------

# Cora & CiteSeer
for ((i=0; i<3; i++))
do
    for ((j=0; j<2; j++))
    do
        echo ${MODELS[j]}
        echo ${DATASETS[i]}
        python tune.py \
        --model ${MODELS[j]} \
        --dataset ${DATASETS[i]} \
        --hard \
        --device cuda:4 \
        --save_prefix graph_tune \
        --iteration 1 \
        --ratio_graph 10 \
        --quant \
        --num_bits 1
    done
done

# Pubmed
for ((i=0; i<2; i++))
do
    echo ${MODELS[i]}
    echo ${DATASETS[2]}
    python tune.py \
    --model ${MODELS[i]} \
    --dataset ${DATASETS[2]} \
    --hard \
    --device cpu \
    --save_prefix graph_tune \
    --iteration 1 \
    --ratio_graph 10 \
    --quant \
    --num_bits 1
done