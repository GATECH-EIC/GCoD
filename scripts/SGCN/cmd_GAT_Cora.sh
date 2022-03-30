PR_RATIO=10

# python train.py \
# --model GAT \
# --dataset Cora \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition

python tune.py \
--model GAT \
--dataset Cora \
--device cuda:4 \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO}