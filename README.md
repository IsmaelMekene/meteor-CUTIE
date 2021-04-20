# :space_invader: myCUTIE
This is a Computer Vision project aiming to create a Convolutional Universal Text Information Extractor from scratch. This is at the time a spatial and semantic segmentation project. The implementation of the model has been inspired from the original  [CUTIE paper](https://arxiv.org/abs/1903.12363v4) admited to CVPR 2019 at the Computer Vision and Pattern Recognition subjects.

## Goal  

The goal of this project is at the be able to predict on a given receipt tickect, a zone of interest (in our case it is mostly `Total amount`)

## Installation 
 
```
pip install -r requirements.txt
pip install segmentation-models
```

## Data, Preprocessing & Processing

### Data

The dataset is consisted of 200 photographic images of receipt, restaurant and also taxi bills. The dataset can be downloaded at [image dataset](https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/)

### Receipt labelling

The initial labelling steps have been done through `google vision ocr` API, in order to store the relative positions of every text areas on each receipt. The raw infos were given back into json format.
With the `data_manipulation.py` functions, the json files were successfully cleaned so that every original receipt has been reshaped and resized into square (keeping the relative positions of the text ares). And more importantly, for a receipt, its corresponding `mask` was generated.
In addition, getting the precised positions of the zones of interest was not an easy task as the receipts picture were note origibally taken in a perfect position. For this case the `IoU` Intersection Over Union was a good trick to make it work. The `IoU` was used to have precision between the whole receipts labelled by `google vision ocr` API and the zones of interest manually labelled with `vgg annotator`.



  
  `image`             |  `mask`             |  `zone of interest`
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1087img.png)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1087mask.png)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/over1087.png)


### Grid

As this project is at the time, a Spatial and Semantic segmentation, It was quite obvious that the classical image segmentation technics would not be perfect. In order to enconter that, the notion of Semantic had to be introduced to make the model more robust in predicting our zone of interest.
The bright ideal of the grid was mentionned in the original  [CUTIE paper](https://arxiv.org/abs/1903.12363v4). The concept is simplify to the fact that:
- The grid would have a propotional size to the original image
- Each text area (`token`) on the original would be represented by its center on the grid
- Each `token` position would then be filled with the corresponding embedding vector


`centers`             |  `Embedding Vectors`
:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/centergrid.png)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1037csv.png)

In the case of this project, the grids were reduced at the quater size of the original image. The `Tokenization` and `Embedding` were made with `GLOVE`: Global Vectors for Word Representation, the compressed files for `GLOVE` can be found [here glove](https://nlp.stanford.edu/projects/glove/).

The work on generated the Embedding vectors for each token on the receipts can be problematic in cases where, there are words on the receipts that are not present into to `GLOVE` Vocabulary. To handle this issue, all the words not into `GLOVE` were reset to a unique `dummy token` and to this `dummy token` has been affected an Embedding vector composed just with 0s.

This is a camembert plot of the project actual vocabulary size proportionally to `GLOVE` vocabulary

<p align="center">
  <img src="https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/ratio.png">
</p>


## Models

The proposed models implemated in the paper [CUTIE paper](https://arxiv.org/abs/1903.12363v4) were several although they all jhave in common the `Pyramidal` structure in addition to the dilated convolutionnal layers.
In the case of this project, two of these `Pyramidal` models were implemented and the training has been done on both in order to evalute the results.

### DenseASPP: Dense Atrous Spatial Pyramid Pooling

The original paper can be found here [denseaspp](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf), it has been admitted to CVPR.

DensASPP simply is named after the backbone model `densenet` and in contrary to it, DenseAspp tkes into account dilation rates.
In this project, attempt had been made to reach a successful performance of DenseASPP. However, it was quite painful, as due to the small amount of data and the lack of pretrained DenseASPP, it was not possible to achieve much with DenseASPP (In the due time!).


`Train_loss`             |  `Valid_loss`
:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/training_loss_densaspp.svg)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/validation_loss_denseaspp.svg)
 
#### predictions

Following the training, although it was obvious that the model would not perform well. a prediction has been made for the sake of the evaluation with PSPNet.

`new_receipt`             |  `prediction`
:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1096raw.png)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/lolpred.png)



### PSPNet: Pyramid Scene Parsing Network

The original paper can be found here [pspnet](https://arxiv.org/pdf/1612.01105.pdf), it has been admitted to CVPR.

PSPNet is a semantic segmentation model that utilises a pyramid parsing module that exploits global context information by different-region based context aggregation. The local and global clues together make the final prediction more reliable. In this project, PSPNet have been implemented and a relatively sucessful performance was reached. As visible on the loss graphs, the model seemed to learn. Although PSPNet is not consistent in term of dilated convolutional layers, pretrained PSPNet were found, thanks to [segmentation-models](https://github.com/qubvel/segmentation_models/).

`Train_loss`             |  `Valid_loss`
:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/training_loss_pspnet.svg)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/validation_loss_pspnet.svg)

#### predictions

Following the training, predictions have been made on a receipt not from the training dataset. It was visivle that the model had learnt to predict the `zone of interest` (In this case `total amount`).

  `new_receipt`             |  `prediction`             |  `reconstruction`
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1096raw.png)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1096pred.png)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1096over.png)

## Results

### Metrics

#### in progress ...

The principal metric used in this project is `MeanIoU` Intersection Over Union.
