PR_RATIO=10

# python train.py \
# --model GAT \
# --dataset Pubmed \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition

python tune.py \
--model GAT \
--dataset Pubmed \
--device cpu \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO}