# GCoD: Graph Convolutional Network Acceleration via Dedicated Algorithm and Accelerator Co-Design

**Haoran You**, Tong Geng, Yongan Zhang, Ang Li, Yingyan Lin (Also credit to Cheng Wan's help on graph paritioning)

Accepted by [HPCA 2022](https://hpca-conf.org/2022/) ([Paper](https://arxiv.org/pdf/2112.11594.pdf) | [Slide](https://github.com/ranery/GCoD/HPCA-GCoD.pdf) | [Youtube](https://www.youtube.com/watch?v=Zx1sMyzwOtY) | [Codebase](https://github.com/ranery/GCoD))

## Overview of the Co-Design Framework

![overview](./figures/overview.png)

## Usage of the Provided Minimalistic Codebase

> Prerequisite

```shell script
conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric
pip install tqdm
pip install ogb
conda install -c dglteam dgl-cuda11.0
```

> Pretrain GCNs on partitioned graphs

````bash
python train.py \
    --model GCN \
    --dataset Cora \
    --partition \
    --device cuda:0 \
    --save_prefix pretrain_partition \
    --quant \
    --enable_chunk_q \
    --num_act_bits 6 \
    --num_wei_bits 6 \
    --num_agg_bits 6
````

More examples are provided in `./scripts/cmd_pretrain.sh`.

Supported models
- GCN
- GAT
- GIN
- GraphSAGE

Supported datasets
- Cora
- CiteSeer
- Pubmed
- NELL

For training on Reddit or larger graphs, please refer to [DeeperGCN](https://github.com/lightaime/deep_gcns_torch) or [BNS-GCN](https://github.com/RICE-EIC/BNS-GCN) and adapt the code accordingly.

> Tuning according to both sparse and polarization/diagonalization regularization terms

````bash
python tune.py \
    --model GCN \
    --dataset Cora \
    --hard \
    --device cuda:4 \
    --save_prefix graph_tune \
    --iteration 1 \
    --ratio_graph 10 \
    --quant \
    --num_bits 16
````

> Visualization of the Resulting Adjacency Matrix

![adj](./figures/adj.png)


## Speedups over Other Platforms

![comp](./figures/comp.png)

## Citation

If you find this codebase useful to your research, please cite:

````
@inproceedings{you2021gcod,
  title={GCoD: Graph Convolutional Network Acceleration via Dedicated Algorithm and Accelerator Co-Design},
  author={You, Haoran and Geng, Tong and Zhang, Yongan and Li, Ang and Lin, Yingyan},
  booktitle={The 28th IEEE International Symposium on High-Performance Computer Architecture (HPCA-28)},
  year={2022}
}
````