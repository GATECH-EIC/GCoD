PR_RATIO=5
BITS=6
CHUNK_MIN=10

# python train_sage.py \
# --model GraphSAGE \
# --dataset CiteSeer \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition \
# --quant \
# --num_act_bits ${BITS} \
# --num_wei_bits ${BITS} \
# --num_agg_bits ${BITS}
# --enable_chunk_q \

python tune_sage.py \
--model GraphSAGE \
--dataset CiteSeer \
--device cuda:1 \
--save_prefix graph_tune_sage \
--iteration 1 \
--ratio_graph ${PR_RATIO} \
--quant \
--num_act_bits ${BITS} \
--num_wei_bits ${BITS} \
--num_agg_bits ${BITS} \
--hard \
--chunk_min ${CHUNK_MIN}
# --enable_chunk_q \