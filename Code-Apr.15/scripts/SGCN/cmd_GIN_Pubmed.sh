PR_RATIO=10

# python train.py \
# --model GIN \
# --dataset Pubmed \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition

python tune.py \
--model GIN \
--dataset Pubmed \
--device cpu \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO}