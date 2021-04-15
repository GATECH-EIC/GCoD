PR_RATIO=10
BITS=6
CHUNK_MIN=40

# python train.py \
# --model GCN \
# --dataset Pubmed \
# --partition \
# --device cuda:0 \
# --save_prefix pretrain_partition \
# --quant \
# --enable_chunk_q \
# --num_act_bits 6 \
# --num_wei_bits 6 \
# --num_agg_bits 6

python tune.py \
--model GCN \
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