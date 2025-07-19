# scDFN: Enhancing Single-cell RNA-seq Clustering with Deep Fusion Networks

## Introduction

Single-cell RNA sequencing enables the measurement of whole-transcriptome gene expression at single-cell resolution. Clustering analysis can reveal cell types and states, providing crucial insights for dissecting cellular heterogeneity in complex tissues. However, many existing methods have limitations: they require predefined numbers of clusters and rely on initial cluster assignments. When processing high-dimensional, sparse, and noisy scRNA-seq data, they struggle to capture the intrinsic patterns and structures of cells. Additionally, insufficient utilization of prior knowledge leads to deviations between clustering results and actual scenarios. To address these issues, we propose the scAClc algorithm, a novel enhanced clustering method for scRNA-seq data analysis. This algorithm identifies key genes through a dual feature selection strategy, combining highly variable genes with important genes evaluated by a random forest model. While effectively reducing dimensionality, it retains core biological information, laying a more accurate feature foundation for subsequent clustering analysis. To enhance the discriminative ability of cell characterization, the algorithm incorporates a contrastive learning mechanism. It explores similarity information between samples through a specially designed loss function, strengthens feature distinguishability between different cell populations, and thus generates more robust low-dimensional embeddings. Furthermore, the algorithm adopts a data-adaptive clustering strategy, which autonomously regulates the cluster merging process by dynamically evaluating intra-class and inter-class distances. It can conform to the intrinsic structure of the data without the need to predefine the number of clusters. Moreover, prior information is integrated as an enhanced constraint during the clustering process to further improve the reliability of clustering results. To validate the performance of scAClc, we compared it with five state-of-the-art algorithms on 15 real scRNA-seq datasets. Experimental results demonstrate that the proposed algorithm outperforms these competing methods. Ablation studies on various modules of the algorithm also confirm that these modules are complementary and can effectively enhance the algorithm's performance. Our method, scAClc, is available at \url{https://github.com/zhengyucai-lib/scAClc}.

## Environment

* Anaconda
* python 3.8+
## Dependency

* Pytorch (2.0+)
* Numpy  1.26.4
* Torchvision 0.17.2
* Matplotlib 3.8.4
* h5py 3.11.0
* Matplotlib 3.8.4
* pandas 2.2.2
* numpy 1.26.4
* scanpy 1.10.1
* scipy 1.12.0

## Installation

1. Download and install Anaconda.
   ```git clone https://github.com/11051911/scDFN.git```

2. Create the prosperousplus environment.

   ```conda create -n scDFN python=3.9.13```

3. Activate the prosperousplus environment and install the dependencies.

   ```conda activate scDFN```

   ```pip install or conda install```

## Usage

Here we provide an implementation of Enhancing single-cell RNA-seq Clustering with Deep Fusion Networks (scDFN) in PyTorch, along with an execution example on the goolam dataset. The repository is organised as follows:

- `scDFN.py`: defines the architecture of the whole network.
- `IGAE.py`: defines the improved graph autoencoder.
- `AE.py`: defines the autoencoder.
- `opt.py`: defines some hyper-parameters.
- `train.py`: the entry point for training and testing.

Finally, `main.py` puts all of the above together and may be used to execute a full training run on goolam.

## Examples: 

### Clustering:
We got the pre-training files in AE, GAE and pretrain respectively, and then trained in the train folder. We provided the pre-training files of goolam.

```python main.py```


## Output:
NMI and ARI values ​​for clustering.

