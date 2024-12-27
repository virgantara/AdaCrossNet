# AdaCrossNet: Adaptive Dynamic Loss Weighting for Cross-Modal Contrastive Point Cloud Learning

#### [Paper Link](https://inass.org/wp-content/uploads/2024/10/2025022911-2.pdf) 

## Introduction

Manual annotation of large-scale point cloud datasets is laborious due to their irregular structure. Whilecross-modal contrastive learning methods such as CrossPoint and CrossNet have progressed in utilizing multimodaldata for self-supervised learning, they still suffer from instability during training caused by the static weighting ofintra-modal (IM) and cross-modal (CM) losses. These static weights fail to account for the varying convergence ratesof different modalities. We propose AdaCrossNet, a novel self-supervised learning framework for point cloudunderstanding that utilizes a dynamic weight adjustment mechanism for IM and CM contrastive learning. AdaCrossNetlearns representations by simultaneously enhancing the alignment between 3-D point clouds and their associated 2D-rendered images within a common latent space. Our dynamic weight adjustment mechanism adaptively balances thecontributions of IM and CM losses during training, guided by the convergence behavior of each modality.

<!-- <img src="docs/CrossNet.jpg" align="center" width="100%"> -->

## Citation

If you entrust our work with value, please consider giving a star ⭐ and citation.

```bibtex
@article{Putra2025,
   author = {Oddy Virgantara Putra and Kohichi Ogata and Eko Mulyanto Yuniarno and Mauridhi Hery Purnomo},
   doi = {10.22266/ijies2025.0229.11},
   issn = {21853118},
   issue = {1},
   journal = {International Journal of Intelligent Engineering and Systems},
   month = {2},
   pages = {134-146},
   title = {AdaCrossNet: Adaptive Dynamic Loss Weighting for Cross-Modal Contrastive Point Cloud Learning},
   volume = {18},
   url = {https://inass.org/wp-content/uploads/2024/10/2025022911-2.pdf},
   year = {2025},
}
```

## Dependencies

Refer `requirements.txt` for the required packages.

## Download data

Datasets are available [here](https://drive.google.com/drive/folders/1dAH9R3XDV0z69Bz6lBaftmJJyuckbPmR?usp=sharing). Run the command below to download all the datasets (ShapeNetRender, ModelNet40, ScanObjectNN, ShapeNetPart) to reproduce the results. Additional [S3DIS](http://buildingparser.stanford.edu/dataset.html) is optional.

```
cd data
source download_data.sh
```

## Train DynamicCrossNet

Refer `python train.py` for the command to train CrossNet.

## Downstream Tasks

### 1. 3D Object Classification 

Run `downstream/classification/main.py`  to perform linear SVM object classification in both ModelNet40 and ScanObjectNN datasets.


### 2. 3D Object Part Segmentation

Refer `downstream/segmentation/main_partseg.py` for fine-tuning experiment for part segmentation in ShapeNetPart dataset.

### 3. 3D Object Semantic Segmentation

Refer `downstream/segmentation/main_semseg.py` for fine-tuning experiment for semantic segmentation in S3DIS dataset.

## Acknowledgements

Our code is heavily borrowed from [CrossNet](https://github.com/liujia99/CrossNet) repository. We thank the authors of CrossNet for releasing their code. 
