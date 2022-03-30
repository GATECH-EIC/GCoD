PR_RATIO=10

# python train_sage.py \
# --model GraphSAGE \
# --dataset CiteSeer \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition

python tune_sage.py \
--model GraphSAGE \
--dataset CiteSeer \
--device cuda:1 \
--save_prefix graph_tune_sage \
--iteration 1 \
--ratio_graph ${PR_RATIO}