# myCUTIE
This is a Computer Vision project aiming to create a Convolutional Universal Text Information Extractor from scratch. This is at the time a spatial and semantic segmentation project. The implementation of the model has been inspired from the original  [CUTIE paper](https://arxiv.org/abs/1903.12363v4) admited to CVPR 2019 at the Computer Vision and Pattern Recognition subjects.

## Readme.md in progress ...

The goal of this project is at the be able to predict on a given receipt tickect, a zone of interest (in our case it is mostly `Total amount`)

## Installation & Usage

```
pip install -r requirements.txt
```

## Data Preprocessing & Processing

### Receipt labelling

The initial labelling steps have been done through `google vision ocr` API, in order to store the relative positions of every text areas on each receipt. The raw infos were given back into json format.
With the `data_manipulation.py` functions, the json files were successfully cleaned so that every original receipt has been reshaped and resized into square (keeping the relative positions of the text ares). And more importantly, for a receipt, its corresponding `mask` was generated.
In addition, getting the precised positions of the zones of interest was not an easy task as the receipts picture were note origibally taken in a perfect position. For this case the `IoU` Intersection Over Union was a good trick to make it work. The `IoU` was used to have precision between the whole receipts labelled by `google vision ocr` API and the zones of interest manually labelled with `vgg annotator`.



  
  `image`             |  `mask`             |  `zone of interest`
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1087img.png)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/1087mask.png)  |  ![](https://github.com/IsmaelMekene/meteor-CUTIE/blob/main/data/over1087.png)


## Results
