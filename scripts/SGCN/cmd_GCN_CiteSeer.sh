PR_RATIO=10

# python train.py \
# --model GCN \
# --dataset CiteSeer \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition \

python tune.py \
--model GCN \
--dataset CiteSeer \
--device cuda:4 \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO}