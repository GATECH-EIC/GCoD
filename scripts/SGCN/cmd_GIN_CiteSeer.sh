PR_RATIO=10

# python train.py \
# --model GIN \
# --dataset CiteSeer \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition

python tune.py \
--model GIN \
--dataset CiteSeer \
--device cuda:8 \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO}