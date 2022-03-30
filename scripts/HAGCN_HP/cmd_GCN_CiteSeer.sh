PR_RATIO=0
# CHUNK_MIN=40 # similar acc
CHUNK_MIN=480 # similar efficiency

# python train.py \
# --model GCN \
# --dataset CiteSeer \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition \

python tune.py \
--model GCN \
--dataset CiteSeer \
--device cuda:0 \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO} \
--hard \
--chunk_min ${CHUNK_MIN}