PR_RATIO=75

# python train.py \
# --model GCN \
# --dataset Cora \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition

python tune.py \
--model GCN \
--dataset Cora \
--device cuda:1 \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO}