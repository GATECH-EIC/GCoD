PR_RATIO=0
CHUNK_MIN=800

# python train.py \
# --model GCN \
# --dataset Pubmed \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition

python tune.py \
--model GCN \
--dataset Pubmed \
--device cpu \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO} \
--hard \
--chunk_min ${CHUNK_MIN}