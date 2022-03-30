PR_RATIO=5
BITS=6
CHUNK_MIN=10

# python train.py \
# --model GIN \
# --dataset Pubmed \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition \
# --quant \
# --enable_chunk_q \
# --num_act_bits ${BITS} \
# --num_wei_bits ${BITS} \
# --num_agg_bits ${BITS}

python tune.py \
--model GIN \
--dataset Pubmed \
--device cpu \
--save_prefix graph_tune \
--iteration 1 \
--ratio_graph ${PR_RATIO} \
--quant \
--enable_chunk_q \
--num_act_bits ${BITS} \
--num_wei_bits ${BITS} \
--num_agg_bits ${BITS} \
--hard \
--chunk_min ${CHUNK_MIN}