# scAClc: A Single-Cell RNA-seq Data Augmentation Clustering Method Based on Adaptive Embedding and Contrastive Learning

## Introduction

Single-cell RNA sequencing enables the measurement of whole-transcriptome gene expression at single-cell resolution. Clustering analysis can reveal cell types and states, providing crucial insights for dissecting cellular heterogeneity in complex tissues. However, many existing methods have limitations: they require predefined numbers of clusters and rely on initial cluster assignments. When processing high-dimensional, sparse, and noisy scRNA-seq data, they struggle to capture the intrinsic patterns and structures of cells. Additionally, insufficient utilization of prior knowledge leads to deviations between clustering results and actual scenarios. To address these issues, we propose the scAClc algorithm, a novel enhanced clustering method for scRNA-seq data analysis. This algorithm identifies key genes through a dual feature selection strategy, combining highly variable genes with important genes evaluated by a random forest model. While effectively reducing dimensionality, it retains core biological information, laying a more accurate feature foundation for subsequent clustering analysis. To enhance the discriminative ability of cell characterization, the algorithm incorporates a contrastive learning mechanism. It explores similarity information between samples through a specially designed loss function, strengthens feature distinguishability between different cell populations, and thus generates more robust low-dimensional embeddings. Furthermore, the algorithm adopts a data-adaptive clustering strategy, which autonomously regulates the cluster merging process by dynamically evaluating intra-class and inter-class distances. It can conform to the intrinsic structure of the data without the need to predefine the number of clusters. Moreover, prior information is integrated as an enhanced constraint during the clustering process to further improve the reliability of clustering results. To validate the performance of scAClc, we compared it with five state-of-the-art algorithms on 15 real scRNA-seq datasets. Experimental results demonstrate that the proposed algorithm outperforms these competing methods. Ablation studies on various modules of the algorithm also confirm that these modules are complementary and can effectively enhance the algorithm's performance. Our method, scAClc, is available at https://github.com/zhengyucai-lib/scAClc.

## Environment

* Anaconda
* python 3.8
## Dependency

* Pytorch 2.4.1+cu121
* Torchvision 0.19.1+cu121
* Matplotlib 3.7.5
* numpy  1.24.3
* h5py  3.9.0
* pandas 1.3.5
* scanpy 1.9.8
* scipy 1.10.1
* scikit-learn 1.2.2
* networkx 2.6.3

## Installation

1. Download and install Anaconda.
   ```git clone https://github.com/zhengyucai-lib/scAClc```

2. Create the prosperousplus environment.

   ```conda create -n scAClc python=3.8.10```

3. Activate the prosperousplus environment and install the dependencies.

   ```conda activate scAClc```

   ```pip install or conda install```


## Quick start

Load the data to be analyzed:

```python
import scanpy as sc

adata = sc.AnnData(data)
```

Perform data pre-processing:

```python
# Basic filtering
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)

adata.raw = adata.copy()

# Total-count normlize, logarithmize the data  
sc.pp.normalize_per_cell(adata)
adata.obs['scale_factor'] = adata.obs.n_counts / adata.obs.n_counts.median()

sc.pp.log1p(adata)
```

Run the scAClc method:

```python
from scaclc import run_scaclc
adata = run_scaclc(adata)
```

The output adata contains cluster labels in `adata.obs['scaclc_cluster']` and the cell embeddings in `adata.obsm['scaclc_emb']`. The embeddings can be used as input of other downstream analyses.

## Output:
NMI and ARI values ​​for clustering.

<ins>Please refer to `scAClc.ipynb` for a detailed description of scAClc's usage.<ins>

