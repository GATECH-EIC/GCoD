PR_RATIO=5

# python train_sage.py \
# --model GraphSAGE \
# --dataset Cora \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition

python tune_sage.py \
--model GraphSAGE \
--dataset Cora \
--device cuda:0 \
--save_prefix graph_tune_sage \
--iteration 1 \
--ratio_graph ${PR_RATIO}